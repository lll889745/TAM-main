import os, torch, cv2, subprocess, shutil, copy
import numpy as np
import nltk
from scipy.optimize import minimize_scalar
from pathlib import Path


try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


_HACI_LAYER_OBJECT = "object"
_HACI_LAYER_ATTRIBUTE = "attribute"
_HACI_LAYER_FUNCTIONAL = "functional"
_HACI_KNOWN_LAYERS = {_HACI_LAYER_OBJECT, _HACI_LAYER_ATTRIBUTE, _HACI_LAYER_FUNCTIONAL}


def _env_flag(name, default=True):
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).lower() not in {"0", "false", "no"}


def _env_float(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


_DEFAULT_INTERFERENCE_MODE = os.environ.get("TAM_INTERFERENCE_MODE", "haci").lower()
_DEFAULT_HACI_LAYER_PRIORS = {
    _HACI_LAYER_OBJECT: _env_float("TAM_HACI_PRIOR_OBJECT", 1.35),
    _HACI_LAYER_ATTRIBUTE: _env_float("TAM_HACI_PRIOR_ATTRIBUTE", 0.85),
    _HACI_LAYER_FUNCTIONAL: _env_float("TAM_HACI_PRIOR_FUNCTIONAL", 0.9),
}
_DEFAULT_HACI_CONFIG = {
    "use_object_attention": _env_flag("TAM_HACI_OBJ_ATT", True),
    "include_attribute_layer": _env_flag("TAM_HACI_ATTR", True),
    "include_functional_layer": _env_flag("TAM_HACI_FUNC", True),
    "use_layer_gating": _env_flag("TAM_HACI_GATE", True),
    "layer_priors": dict(_DEFAULT_HACI_LAYER_PRIORS),
    "max_interference_scale": _env_float("TAM_HACI_MAX_SCALE", 3.0),
    "residual_ratio": _env_float("TAM_HACI_RESIDUAL", 0.5),
    "object_gain": _env_float("TAM_HACI_OBJECT_GAIN", 1.35),
    "layer_focus_gain": _env_float("TAM_HACI_LAYER_FOCUS", 1.25),
    "functional_focus_gain": _env_float("TAM_HACI_FUNC_FOCUS", 1.35),
}


_HAS_XELATEX = shutil.which("xelatex") is not None
_WARNED_XELATEX = False


def _ensure_nltk_pos_tagger():
    """Ensure NLTK taggers required for HACI are present."""

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger", quiet=True)
    try:
        nltk.data.find("help/tagsets/universal_tagset.pickle")
    except LookupError:
        nltk.download("universal_tagset", quiet=True)


def _tokens_to_words(tokens):
    """Group subword tokens into coarse words for POS tagging."""

    words, groups = [], []
    buffer_tokens, buffer_pieces = [], []

    def flush():
        if not buffer_tokens:
            return
        word = "".join(buffer_pieces).strip()
        if word:
            words.append(word)
            groups.append(list(buffer_tokens))
        buffer_tokens.clear()
        buffer_pieces.clear()

    for idx, token in enumerate(tokens):
        normalized = token.replace("▁", " ").replace("Ġ", " ").replace("Ċ", " ")
        starts_new = token.startswith(("▁", "Ġ")) or token == "Ċ"
        if starts_new and buffer_tokens:
            flush()
        if not buffer_tokens:
            buffer_tokens.append(idx)
            buffer_pieces.append(normalized)
        else:
            buffer_tokens.append(idx)
            buffer_pieces.append(normalized)
    flush()

    return words, groups


def _assign_layers(tokens):
    """Assign HACI layers to each token via POS tagging."""

    layers = [_HACI_LAYER_FUNCTIONAL for _ in tokens]
    words, groups = _tokens_to_words(tokens)
    if not words:
        return layers

    _ensure_nltk_pos_tagger()
    try:
        tagged = nltk.pos_tag(words, tagset="universal")
    except LookupError:
        tagged = [(w, "NOUN") for w in words]

    pos_to_layer = {
        "NOUN": _HACI_LAYER_OBJECT,
        "PROPN": _HACI_LAYER_OBJECT,
        "ADJ": _HACI_LAYER_ATTRIBUTE,
        "VERB": _HACI_LAYER_ATTRIBUTE,
        "ADV": _HACI_LAYER_ATTRIBUTE,
        "NUM": _HACI_LAYER_ATTRIBUTE,
        "PRON": _HACI_LAYER_FUNCTIONAL,
        "DET": _HACI_LAYER_FUNCTIONAL,
        "ADP": _HACI_LAYER_FUNCTIONAL,
        "AUX": _HACI_LAYER_FUNCTIONAL,
        "PART": _HACI_LAYER_FUNCTIONAL,
        "CONJ": _HACI_LAYER_FUNCTIONAL,
        "SCONJ": _HACI_LAYER_FUNCTIONAL,
        "CCONJ": _HACI_LAYER_FUNCTIONAL,
        "PUNCT": _HACI_LAYER_FUNCTIONAL,
        "INTJ": _HACI_LAYER_FUNCTIONAL,
        "X": _HACI_LAYER_FUNCTIONAL,
    }

    for (_, pos), token_indices in zip(tagged, groups):
        layer = pos_to_layer.get(pos, _HACI_LAYER_FUNCTIONAL)
        for idx in token_indices:
            if 0 <= idx < len(layers):
                layers[idx] = layer

    return layers


def _object_attention_interference(maps, weights):
    stacked = np.stack(maps, axis=0)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if weights.sum() <= 1e-8:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / (weights.sum() + 1e-8)

    stacked_norm = stacked / (stacked.max(axis=1, keepdims=True) + 1e-6)
    conv1 = stacked_norm * weights[:, None]
    conv2 = np.maximum(conv1, 0.0)
    logits = conv2 - conv2.max(axis=0, keepdims=True)
    attn = np.exp(logits)
    attn = attn / (attn.sum(axis=0, keepdims=True) + 1e-8)
    return (attn * stacked).sum(axis=0)


def _linear_interference(maps, weights):
    stacked = np.stack(maps, axis=0)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if weights.sum() <= 1e-8:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / (weights.sum() + 1e-8)
    return (stacked * weights[:, None]).sum(axis=0)


def _compute_haci_interference(txt_tokens, txt_scores, img_store, current_idx, haci_cfg, target_layer=None):
    if current_idx >= len(txt_tokens):
        return None

    target_token = txt_tokens[current_idx]
    candidate_indices = []
    for idx in range(current_idx):
        if idx >= len(img_store):
            break
        entry = img_store[idx]
        if entry is None:
            continue
        if idx < len(txt_tokens) and txt_tokens[idx] == target_token:
            continue
        candidate_indices.append(idx)

    if not candidate_indices:
        return None

    layer_maps = {_HACI_LAYER_OBJECT: [], _HACI_LAYER_ATTRIBUTE: [], _HACI_LAYER_FUNCTIONAL: []}
    layer_weights = {_HACI_LAYER_OBJECT: [], _HACI_LAYER_ATTRIBUTE: [], _HACI_LAYER_FUNCTIONAL: []}

    if len(txt_scores) < len(txt_tokens):
        txt_scores = np.pad(txt_scores, (0, len(txt_tokens) - len(txt_scores)), constant_values=0.0)

    for idx in candidate_indices:
        entry = img_store[idx]
        if entry is None or entry.get("map") is None:
            continue
        layer = entry.get("layer", _HACI_LAYER_FUNCTIONAL)
        if layer not in _HACI_KNOWN_LAYERS:
            layer = _HACI_LAYER_FUNCTIONAL
        if layer == _HACI_LAYER_ATTRIBUTE and not haci_cfg.get("include_attribute_layer", True):
            continue
        if layer == _HACI_LAYER_FUNCTIONAL and not haci_cfg.get("include_functional_layer", True):
            continue
        weight = float(txt_scores[idx]) if idx < len(txt_scores) else 0.0
        layer_maps[layer].append(entry["map"])
        layer_weights[layer].append(max(weight, 0.0))

    interference_components = {}
    for layer, maps in layer_maps.items():
        if not maps:
            continue
        weights = layer_weights[layer]
        if layer == _HACI_LAYER_OBJECT:
            if haci_cfg.get("use_object_attention", True):
                interference_components[layer] = _object_attention_interference(maps, weights)
            else:
                interference_components[layer] = _linear_interference(maps, weights)
        else:
            interference_components[layer] = _linear_interference(maps, weights)

    if not interference_components:
        return None

    object_gain = max(haci_cfg.get("object_gain", 1.0), 0.0)
    if object_gain != 1.0 and _HACI_LAYER_OBJECT in interference_components:
        interference_components[_HACI_LAYER_OBJECT] = interference_components[_HACI_LAYER_OBJECT] * object_gain

    strengths = {layer: sum(layer_weights[layer]) for layer in interference_components}
    layer_priors = haci_cfg.get("layer_priors", {}) or {}
    weighted_strengths = {}
    for layer, strength in strengths.items():
        prior = layer_priors.get(layer, 1.0)
        weighted_strengths[layer] = max(strength, 0.0) * max(prior, 0.0)

    available_layers = list(interference_components.keys())
    if haci_cfg.get("use_layer_gating", True):
        total_weight = sum(weighted_strengths.get(layer, 0.0) for layer in available_layers)
        if total_weight <= 1e-8:
            gating = {layer: 1.0 / len(available_layers) for layer in available_layers}
        else:
            gating = {layer: weighted_strengths.get(layer, 0.0) / (total_weight + 1e-8) for layer in available_layers}
    else:
        fixed_weights = {layer: max(layer_priors.get(layer, 1.0), 0.0) for layer in available_layers}
        total_weight = sum(fixed_weights.values())
        if total_weight <= 1e-8:
            gating = {layer: 1.0 / len(available_layers) for layer in available_layers}
        else:
            gating = {layer: fixed_weights[layer] / (total_weight + 1e-8) for layer in available_layers}

    focus_gain = max(haci_cfg.get("layer_focus_gain", 1.0), 0.0)
    target_focus_gain = focus_gain
    if target_layer == _HACI_LAYER_FUNCTIONAL:
        target_focus_gain = max(target_focus_gain, haci_cfg.get("functional_focus_gain", 1.0))
    if target_focus_gain > 1.0 and target_layer in gating:
        gating[target_layer] *= target_focus_gain
        norm = sum(gating.values())
        if norm > 1e-8:
            for layer in gating:
                gating[layer] /= norm

    template = next(iter(interference_components.values()))
    total_interference = np.zeros_like(template)
    for layer, comp in interference_components.items():
        total_interference += gating.get(layer, 0.0) * comp

    return total_interference


def _compute_eci_interference(txt_tokens, txt_scores, img_store, current_idx):
    if current_idx >= len(txt_tokens):
        return None

    target_token = txt_tokens[current_idx]
    maps = []
    weights = []
    if len(txt_scores) < len(txt_tokens):
        txt_scores = np.pad(txt_scores, (0, len(txt_tokens) - len(txt_scores)), constant_values=0.0)

    for idx in range(current_idx):
        entry = img_store[idx]
        if entry is None or entry.get("map") is None:
            continue
        if idx < len(txt_tokens) and txt_tokens[idx] == target_token:
            continue
        maps.append(entry["map"])
        weights.append(max(float(txt_scores[idx]), 0.0))

    if not maps:
        return None

    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 1e-8:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / (weights.sum() + 1e-8)

    return (np.stack(maps, axis=0) * weights[:, None]).sum(axis=0)


def _compute_interference(mode, txt_tokens, txt_scores, img_store, current_idx, haci_cfg, target_layer=None):
    mode = (mode or "").lower()
    if mode not in {"haci", "eci", "none"}:
        mode = "haci"
    if mode == "none":
        return None
    if mode == "eci":
        return _compute_eci_interference(txt_tokens, txt_scores, img_store, current_idx)
    return _compute_haci_interference(txt_tokens, txt_scores, img_store, current_idx, haci_cfg, target_layer=target_layer)


def rank_guassian_filter(img, kernel_size=3):
    """
    Apply a rank-based Gaussian-weighted filter for robust activation map denoising.

    Parameters:
    img : np.ndarray
        Input 2D grayscale image.
    kernel_size : int
        Size of the square kernel (must be odd).

    Returns:
    filtered_img : np.ndarray
        Denoised image after applying the Gaussian weighted rank filter.

    Note:
        The sigma (std) of is refined to coefficient of variation for robust results
    """

    filtered_img = np.zeros_like(img)
    pad_width = kernel_size // 2
    padded_img = np.pad(img, pad_width, mode='reflect')
    ax = np.array(range(kernel_size ** 2)) - kernel_size ** 2 // 2

    for i in range(pad_width, img.shape[0] + pad_width):
        for j in range(pad_width, img.shape[1] + pad_width):
            window = padded_img[i - pad_width:i + pad_width + 1,
                                j - pad_width:j + pad_width + 1]

            sorted_window = np.sort(window.flatten())
            mean = sorted_window.mean()
            if mean > 0:
                sigma = sorted_window.std() / mean # std -> cov
                kernel = np.exp(-(ax**2) / (2 * sigma**2))
                kernel = kernel / np.sum(kernel)
                value = (sorted_window * kernel).sum()
            else:
                value = 0
            filtered_img[i - pad_width, j - pad_width] = value
    
    return filtered_img


def least_squares(map1, map2):
    """
    Find the scalar that minimizes the squared difference between map1 and scalar * map2.

    Args:
        map1 (np.ndarray): First data array.
        map2 (np.ndarray): Second data array.

    Returns:
        float: Optimal scalar multiplier.
    """

    def diff(x, map1, map2):
        return np.sum((map1 - map2 * x)**2)

    result = minimize_scalar(diff, args=(map1, map2))
    return result.x


def generate_latex(words, relevances, cmap="bwr", font=r'{18pt}{21pt}'):
    """
    Generate LaTeX code to visualize tokens with colored backgrounds or text, based on their relevance scores.

    Args:
        words (list of str): List of token strings, where tokens starting with '▁' or 'Ġ' represent spaces.
        relevances (list of float): List of relevance scores corresponding to each token.
            - relevance >= 0: earlier context tokens, color-coded with a jet colormap.
            - relevance == -1: current explained token, shown with black background and white text.
            - relevance == -2: next tokens, rendered in gray color.
            - relevance == -3: special marker to add a newline and "Candidates:" label.
            - relevance == -4: special marker to add a newline and print the word string as is.
        cmap (str): Colormap to use for positive relevances (default "bwr" - unused in current code).
        font (str): Font size and line spacing in LaTeX format, e.g. '{18pt}{21pt}'.

    Returns:
        str: A complete LaTeX document as a string with colored tokens visualized.
    """


    latex_code = r'''
    \documentclass[arwidth=200mm]{standalone}
    \renewcommand{\normalsize}{\fontsize''' + font + r'''\selectfont}
    \usepackage[dvipsnames]{xcolor}

    \begin{document}
    \fbox{
    \parbox{\textwidth}{
    \setlength\fboxsep{0pt}
    '''

    for i in range(len(words)):
        word = words[i]
        relevance = relevances[i]

        # relevance >= 0 for earlier context tokens (jet colors)
        if relevance >= 0:
            jet_colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
            b, g, r = jet_colormap[int(relevances[i] * 255)][0].tolist()
            if word[:2] == '$ ' and word[-1] == '$': # candidates
                latex_code += f' \\textbf{{\\textcolor[RGB]{{{r},{g},{b}}}{{\\strut {word}}}}}, '
            elif word.startswith('▁') or word.startswith('Ġ') or word.startswith(' '):
                word = word.replace('▁', ' ').replace('Ġ', ' ')
                latex_code += f' \\textbf{{\\textcolor[RGB]{{{r},{g},{b}}}{{\\strut {word}}}}}'
            else:
                latex_code += f'\\textbf{{\\textcolor[RGB]{{{r},{g},{b}}}{{\\strut {word}}}}}'

        # for current explained token (black)
        elif relevance == -1:
            if word.startswith('▁') or word.startswith('Ġ') or word.startswith(' '):
                word = word.replace('▁', ' ').replace('Ġ', ' ')
                latex_code += f' \\textbf{{\\colorbox[RGB]{{{0},{0},{0}}}{{\\textcolor[RGB]{{{255},{255},{255}}}{{\\strut {word}}}}}}}'
            else:
                latex_code += f'\\textbf{{\\colorbox[RGB]{{{0},{0},{0}}}{{\\textcolor[RGB]{{{255},{255},{255}}}{{\\strut {word}}}}}}}'

        # for next tokens (gray)
        elif relevance == -2:
            b, g, r = 200, 200, 200
            if word.startswith('▁') or word.startswith('Ġ') or word.startswith(' '):
                word = word.replace('▁', ' ').replace('Ġ', ' ')
                latex_code += f' \\textbf{{\\textcolor[RGB]{{{r},{g},{b}}}{{\\strut {word}}}}}'
            else:
                latex_code += f'\\textbf{{\\textcolor[RGB]{{{r},{g},{b}}}{{\\strut {word}}}}}'

        # for top pred
        elif relevance == -3:
            latex_code += '\\\\$Candidates:$'

        # for custom vis str
        elif relevance == -4:
            latex_code += '\\\\' + word

    latex_code += r'}}\end{document}'

    return latex_code


def compile_latex_to_jpg(latex_code, path='word_colors.pdf', delete_aux_files=True, dpi=500):
    """
    Compile a LaTeX string into a JPG image.

    Parameters:
    - latex_code (str): The LaTeX source code to compile.
    - path (str or Path): File path for intermediate PDF and auxiliary files. The output image is returned as an array.
    - delete_aux_files (bool): Whether to delete auxiliary files (.aux, .log, .tex, .pdf) after compilation.
    - dpi (int): Resolution for the output image in dots per inch.

    Returns:
    - img (numpy.ndarray): The compiled LaTeX rendered as a color image (BGR) array.
                          Returns None if compilation fails.
    """

    global _WARNED_XELATEX
    if not _HAS_XELATEX or fitz is None:
        if not _WARNED_XELATEX:
            if not _HAS_XELATEX:
                print('Skip text visualization, xelatex is not available on this system.')
            else:
                print('Skip text visualization, PyMuPDF (fitz) is not available on this system.')
            _WARNED_XELATEX = True
        return None

    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)

    with open(path.with_suffix(".tex"), 'w') as f:
        f.write(latex_code)

    try:
        res_code = subprocess.run(['xelatex', '--output-directory', path.parent, path.with_suffix(".tex")], \
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    except:
        print('Skip, fail to compile: ' + res_code)
        return None

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    page = fitz.open(path.with_suffix(".pdf")).load_page(0)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    if delete_aux_files:
        for suffix in ['.aux', '.log', '.tex', '.pdf']:
            os.remove(path.with_suffix(suffix))

    getpngdata = pix.tobytes("png")
    image_array = np.frombuffer(getpngdata, dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_ANYCOLOR)[:,:,:3]
    return img


def vis_text(words, relevances, candidates, candi_scores, vis_token_idx, path='heatmap.jpg', font=r'{18pt}{21pt}'):
    """
    Visualizes text tokens and their relevance scores as a heatmap image using LaTeX.

    This function processes a list of words and their corresponding relevance scores, along with candidate tokens 
    and their scores, to create a color-coded heatmap visualization. It handles special LaTeX characters by escaping 
    them appropriately to ensure correct LaTeX rendering. The visualization includes the explained tokens, subsequent 
    tokens, and top prediction candidates with distinct coloring based on their scores.

    Args:
        words: All tokens need to visualize.
        relevances: Relevance scores corresponding to each token.
        candidates: Candidate tokens (top k predictions).
        candi_scores: Scores associated with each candidate token.
        vis_token_idx (int): Index of the token to vis (explain).
        path (str, optional): File path to save the generated heatmap image. Defaults to 'heatmap.jpg'.
        font (str, optional): LaTeX font size settings for the visualization. Defaults to r'{18pt}{21pt}'.

    Returns:
        str: Numpy image for the visualized texts
    """


    # add scores (-2, gray) for next tokens after the exaplained one
    add_scores = []
    for i in range(len(relevances), len(words[:-1])):
        add_scores.append(-2)

    # explained tokens + next tokens + top pred candidates (see defination of scores in generate_latex)
    all_scores = relevances.tolist() + add_scores + [-3] + candi_scores.cpu().float().tolist()
    all_scores[vis_token_idx] = -1

    # scores correspond to the words
    all_words = words[:-1] + [''] + ['$ ' + _ + '$' for _ in candidates]

    # replace special texts to fit latex
    all_words = [_.replace('\\', '\\backslash').replace('\n', '\\newline').replace('_', '\\_').replace('^', '\\^').replace('&', '\\&').replace('%', '\\%').replace('Ċ', '\\newline') for _ in all_words]

    # to latex, then to img
    latex_code = generate_latex(all_words, all_scores, cmap='bwr', font=font)
    return compile_latex_to_jpg(latex_code, path=path, delete_aux_files=True)


def multimodal_process(raw_img, vision_shape, img_scores, txt_scores, txts, candidates, candi_scores, \
                       vis_token_idx, img_save_fn, eval_only=False, vis_width=-1):
    """
    Process multimodal tokens: visualizing combined image and text activations with normalizing, filtering, and blending scores.

    This function processes image and text token scores to generate a multimodal visualization:
    - Normalizes image and text token scores together for comparability.
    - Applies the Rank Gank Guassian Filter for vision tokens.
    - Visualizes text token via latex.
    - Combines visual maps of image and text tokens for final output.
    - Supports single image, multiple images, and video batch inputs.
    - Optionally returns only evaluation maps without visualization.

    Args:
        raw_img (np.ndarray or list of np.ndarray): Raw input image(s). For multiple images, provide a list.
        vision_shape (tuple or list of tuples): Shape(s) of vision tokens (height, width) or batch size + shape for video.
        img_scores (np.ndarray): Activation scores for image tokens.
        txt_scores (np.ndarray): Activation scores for text tokens.
        txts (list): Visualized texts, including texts before the target and next words.
        candidates (list): Candidate topK predictions of the explianed token.
        candi_scores (np.ndarray): Scores for candidate tokens.
        vis_token_idx (list): Index of the explained token in all_text to visualize.
        img_save_fn (str): Path to save the visualization image.
        eval_only (bool, optional): If True, only returns evaluation score maps without visualization. Defaults to False.
        vis_width (int, optional): Width for resizing images and visualizations. If -1, no resizing is done. Defaults to -1.

    Returns:
        tuple:
            - out_img (np.ndarray or None): Final blended visualization image combining image and text scores.
            - img_map (np.ndarray or list of np.ndarray): Evaluation score maps for image tokens.
    """


    # normalize multimodal tokens
    txt_scores = txt_scores[:-1] # ignore self score
    all_scores = np.concatenate([img_scores, txt_scores], 0)
    all_scores = (all_scores - all_scores.min()) / (all_scores.max() - all_scores.min())
    img_scores = all_scores[:len(img_scores)]
    txt_scores = all_scores[len(img_scores):]

    eval_only = True if img_save_fn == "" else False

    # for multiple imgs
    if isinstance(vision_shape[0], tuple):
        resized_img, img_map = [], []
        start_idx = 0
        for n in range(len(vision_shape)):
            t_h, t_w = vision_shape[n]
            h, w, c = raw_img[n].shape

            # for fix height
            if vis_width > 0:
                h = int(vis_width)
                w = int(float(w) / h * vis_width)

            # apply the rank_guassian_filter for vision tokens of each img
            end_idx = start_idx + int(t_h * t_w)
            img_map_ = rank_guassian_filter(img_scores[start_idx: end_idx].reshape(t_h, t_w), 3)
            start_idx = end_idx
            img_map_ = (img_map_ * 255).astype('uint8')

            # resize map and raw img if need vis
            if not eval_only:
                img_map_ = cv2.applyColorMap(img_map_, cv2.COLORMAP_JET)
                img_map_ = cv2.resize(img_map_, (w, h))
                if vis_width > 0:
                    raw_img_ = cv2.resize(raw_img[n], (w, h))
                    resized_img.append(raw_img_)

            img_map.append(img_map_)

        # eval only output
        if eval_only:
            return None, img_map

        out_img = [img_map[i] * 0.5 + resized_img[i] * 0.5 for i in range(len(vision_shape))]
        out_img = np.concatenate(out_img, 1)

        # text vis via latex
        try:
            txt_map = vis_text(txts, txt_scores, candidates, candi_scores, vis_token_idx, path=img_save_fn, font=r'{5pt}{6pt}')
        except Exception:
            if _HAS_XELATEX:
                print('Skip text visualization, please check the installation of texlive-xetex.')
            return out_img, img_map

        if not isinstance(txt_map, np.ndarray):
            if _HAS_XELATEX:
                print('Skip txt visualization, please check weather the text special character compatible with LaTeX.')
            return out_img, img_map

        # concat multimodal vis
        txt_map = cv2.resize(txt_map, (out_img.shape[1], int(float(txt_map.shape[0]) / float(txt_map.shape[1]) * out_img.shape[1])))
        out_img = np.concatenate([out_img, txt_map], 0)

        return out_img, img_map

    # single img
    elif len(vision_shape) == 2:
        # set img size
        t_h, t_w = vision_shape
        h, w, c = raw_img.shape
        if vis_width > 0:
            h = int(float(h) / w * vis_width)
            w = int(vis_width)

        # apply filter
        img_scores = rank_guassian_filter(img_scores.reshape(t_h, t_w), 3)
        img_scores = (img_scores * 255).astype('uint8')

        if eval_only:
            return None, img_scores

        img_map = cv2.applyColorMap(img_scores, cv2.COLORMAP_JET)
        img_map = cv2.resize(img_map, (w, h))
        if vis_width > 0:
            raw_img = cv2.resize(raw_img, (w, h))
        out_img = img_map * 0.5 + raw_img * 0.5

        # vis text via latex
        try:
            txt_map = vis_text(txts, txt_scores, candidates, candi_scores, vis_token_idx, path=img_save_fn)
        except Exception:
            if _HAS_XELATEX:
                print('Skip text visualization, please check the installation of texlive-xetex.')
            return out_img, img_scores

        if not isinstance(txt_map, np.ndarray):
            if _HAS_XELATEX:
                print('Skip txt visualization, please check weather the text special character compatible with LaTeX.')
            return out_img, img_scores

        txt_map = cv2.resize(txt_map, (w, int(float(txt_map.shape[0]) / float(txt_map.shape[1]) * w)))
        out_img = np.concatenate([out_img, txt_map], 0)

        return out_img, img_scores

    # video
    else:
        b, t_h, t_w = vision_shape
        h, w, c = raw_img[0].shape
        if vis_width > 0:
            h = int(float(h) / w * vis_width)
            w = int(vis_width)

        img_scores = np.array([rank_guassian_filter(_.reshape(t_h, t_w), 3) for _ in np.array_split(img_scores, b)])
        img_scores = (img_scores * 255).astype('uint8')

        if eval_only:
            return None, img_scores

        img_map = [cv2.resize(cv2.applyColorMap(_, cv2.COLORMAP_JET), (w, h)) for _ in img_scores]
        if vis_width > 0:
            raw_img = [cv2.resize(_, (w, h)) for _ in raw_img]
        out_img = [img_map[i] * 0.5 + raw_img[i] * 0.5 for i in range(b)]
        out_img = np.concatenate(out_img, 1)

        # vis text via latex
        try:
            txt_map = vis_text(txts, txt_scores, candidates, candi_scores, vis_token_idx, path=img_save_fn, font=r'{5pt}{6pt}')
        except Exception:
            if _HAS_XELATEX:
                print('Skip text visualization, please check the installation of texlive-xetex.')
            return out_img, img_scores

        if not isinstance(txt_map, np.ndarray):
            if _HAS_XELATEX:
                print('Skip txt visualization, please check weather the text special character compatible with LaTeX.')
            return out_img, img_scores

        txt_map = cv2.resize(txt_map, (int(w * b), int(float(txt_map.shape[0]) / float(txt_map.shape[1]) * w * b)))
        out_img = np.concatenate([out_img, txt_map], 0)

        return out_img, img_scores



def id2idx(inp_id, target_id, return_last=False):
    """
    Convert a target ID or sequence of IDs to the corresponding index in the input list.

    Args:
        input_ids (list of int): The list of token IDs to search within.
        target_id (int or list of int): The target token ID or sequence of token IDs to find.
        return_last (bool): If True and target_id is a list, return the index of the last token in the matched sequence.
                            Otherwise, return the index of the first token.

    Returns:
        int: The index of the target ID (or start/end of the sequence) in input_ids, or -1 if not found.
    """

    # use a array of tokens as the identifier
    if isinstance(target_id, list):
        n = len(target_id)
        indexes = [i for i in range(len(inp_id) - n + 1) if inp_id[i:i+n] == target_id]
        if len(indexes) > 0:
            # get the idx of the first token as the end identifier
            idx = indexes[-1]

            # get the idx of the last token as the begain identifier
            if return_last:
                idx += len(target_id) - 1
        else:
            idx = -1

    # if the id is unique, use a int is simple
    else:
        try:
            idx = inp_id.index(target_id)
        except:
            idx = -1
    return idx



def TAM(tokens, vision_shape, logit_list, special_ids, vision_input, \
    processor, save_fn, target_token, img_scores_list, eval_only=False,
    interference_mode=None, haci_config=None):

    """
    Generate a Token Activation Map (TAM) with configurable causal interference modeling and
    Rank Guassian Filter for high quality MLLM visual explaination.

    Args:
        tokens (list): The token sequence including input and generated tokens.
        vision_shape (tuple or list): Shape information of the vision input (image/video).
        logit_list (list of torch.Tensor): List of logits tensors for each generation round; 
        special_ids (dict): Dictionary containing special token ids:
            - 'img_id': list of ids to locate the start and end of vision inputs.
              Note: a int value for img_id indicates all tokens of this id.
            - 'prompt_id': tuple of (start_id, end_id) for prompt text tokens.
            - 'answer_id': tuple of (start_id, end_id) for answer tokens.
            Note: 1. The format is [int/list for start, int/list for end].
                  2. The select tokens are [start + 1: end].
                  3. The start list uses the idx of last token, while end uses the first.
        vision_input (array or list): Raw vision input (images or video frames).
        processor: The model processor to convert tokens to text.
        save_fn (str): File path to save the visualization image (optional).
        target_token (int or tuple): The token index or (round_idx, prompt_token_idx) to explain.
        img_scores_list (list): Mutable list used to cache per-token activation maps and layers.
            Note: start with an empty list for the first round of each example.
        eval_only (bool): Whether to run in evaluation mode (affects visualization size).
        interference_mode (str): 'haci' (default), 'eci', or 'none' to control interference modeling.
        haci_config (dict): Optional overrides for HACI behaviour (use_object_attention,
            include_attribute_layer, include_functional_layer, use_layer_gating).

    Returns:
        img_map (np.ndarray): The TAM for eval.

    Workflow:
    1. Convert tokens to list and identify indices for image, prompt, and answer tokens.
    2. Decode prompt and answer tokens into text tokens using the processor.
    3. Determine the target token indices and generation round.
    4. For round 0, recursively process all prompt tokens to generate maps.
    5. Extract the logits for the target token's predicted class and compute relevance scores 
       over prompt, answer, and image tokens.
    6. Apply the selected interference estimator to remove structured interference from earlier context tokens.
    7. Prepare vision input images or frames for visualization.
    8. Identify top candidate tokens to provide context in visualization.
    9. Call multimodal_process to generate the visual explanation map (TAM).
       This step includes the Rank Guassian Filter.
    10. Save the resulting visualization image if a save path is provided.
    11. Return the computed image activation map.

    """

    # start and end id for img, prompt and answer
    img_id = special_ids['img_id']
    prompt_id = special_ids['prompt_id'] # prompt text, start and end id
    answer_id = special_ids['answer_id'] # number of tokens between prompt and answer
    
    # if img_id is a int, take all tokens same to this id
    if len(img_id) == 1:
        img_idx = (np.array(tokens) == img_id[0]).nonzero()[0]
    else:
        img_idx = [id2idx(tokens, img_id[0], True), id2idx(tokens, img_id[1])]

    # convert vocab id to idx in tokens
    prompt_idx = [id2idx(tokens, prompt_id[0], True), id2idx(tokens, prompt_id[1])]
    answer_idx = [id2idx(tokens, answer_id[0], True), id2idx(tokens, answer_id[1])]

    # decode ids

    prompt = processor.tokenizer.tokenize(processor.batch_decode([tokens[prompt_idx[0] + 1: prompt_idx[1]]], \
            skip_special_tokens=False, clean_up_tokenization_spaces=False)[0])
    answer = processor.tokenizer.tokenize(processor.batch_decode([tokens[answer_idx[0] + 1:]], \
            skip_special_tokens=False, clean_up_tokenization_spaces=False)[0])
    txt_all = prompt + answer
    token_layers = _assign_layers(txt_all)

    if interference_mode is None:
        interference_mode = _DEFAULT_INTERFERENCE_MODE
    haci_cfg = copy.deepcopy(_DEFAULT_HACI_CONFIG)
    if haci_config:
        custom_priors = haci_config.get("layer_priors") if isinstance(haci_config, dict) else None
        if custom_priors:
            base_priors = haci_cfg.get("layer_priors", {}).copy()
            base_priors.update(custom_priors)
            haci_cfg["layer_priors"] = base_priors
        for key, value in haci_config.items():
            if key == "layer_priors":
                continue
            haci_cfg[key] = value
    is_haci_mode = (interference_mode == "haci")
    if not is_haci_mode:
        haci_cfg["max_interference_scale"] = float("inf")
        haci_cfg["residual_ratio"] = 0.0
        haci_cfg["use_layer_gating"] = False
        haci_cfg["layer_priors"] = {_HACI_LAYER_OBJECT: 1.0, _HACI_LAYER_ATTRIBUTE: 1.0, _HACI_LAYER_FUNCTIONAL: 1.0}
        haci_cfg["object_gain"] = 1.0
        haci_cfg["layer_focus_gain"] = 1.0
        haci_cfg["functional_focus_gain"] = 1.0

    # round_idx indicates the round of generation, this_token_idx is for the exaplained target token
    round_idx = -1
    this_token_idx = 0

    # for non-first rounds
    if isinstance(target_token, int):
        round_idx = target_token
        this_token_idx = -1 # last token of each answer round
        vis_token_idx = len(prompt) + target_token

    # for the first round, which contrains multiple prompt tokens to explain
    else:
        round_idx, prompt_token_idx = target_token
        this_token_idx = prompt_idx[0] + prompt_token_idx + 1
        vis_token_idx = prompt_token_idx

    # vis prompt tokens at round 0
    if round_idx == 0 and isinstance(target_token, int):
        for t in range(len(prompt) + 1):
            # recursion to process prompt tokens
            img_map = TAM(tokens, vision_shape, logit_list, special_ids, vision_input, processor, \
                          save_fn if t == len(prompt) else '', [0, t], img_scores_list, eval_only,
                          interference_mode=interference_mode, haci_config=haci_config)

            ## the first prompt token is used to reflect the differenec of activation degrees
            if t == 0:
                first_ori = img_map

        return first_ori

    # assign class id
    if round_idx == 0:

        # last token of round 0 is the first generated token
        if prompt_token_idx == len(prompt):
            this_token_idx = logit_list[0].shape[1] - 1
            cls_id = tokens[this_token_idx]

        # record the first prompt with greedy search
        elif prompt_token_idx == 0:
            cls_id = logit_list[0][0, prompt_idx[0] + 1].argmax(0)

        # other maps prompt tokens
        else:
            cls_id = tokens[this_token_idx]

    # generated tokens (round >= 1)
    else:
        cls_id = tokens[answer_idx[0] + round_idx + 1]

    # class activation map from logits of the target token class
    scores = torch.cat([logit_list[_][0, :, cls_id] for _ in range(round_idx + 1)], -1).clip(min=0)

    # get relevance scores
    scores = scores.detach().cpu().float().numpy()
    prompt_scores = scores[prompt_idx[0] + 1: prompt_idx[1]]
    last_prompt = scores[logit_list[0].shape[1] - 1: logit_list[0].shape[1]]
    answer_scores = scores[answer_idx[0] + 1:]
    txt_scores = np.concatenate([prompt_scores, last_prompt, answer_scores], -1)
    if isinstance(img_idx, list):
        img_scores = scores[img_idx[0] + 1: img_idx[1]]
    else:
        img_scores = scores[img_idx]

    img_scores = np.array(img_scores, copy=True)
    raw_img_scores = img_scores.copy()
    layer_idx = min(vis_token_idx, len(token_layers) - 1) if token_layers else 0
    current_layer = token_layers[layer_idx] if token_layers else _HACI_LAYER_FUNCTIONAL

    while len(img_scores_list) <= vis_token_idx:
        img_scores_list.append(None)

    interference = None
    if vis_token_idx < len(txt_all):
        interference = _compute_interference(
            interference_mode,
            txt_all,
            txt_scores,
            img_scores_list,
            vis_token_idx,
            haci_cfg,
            target_layer=current_layer,
        )
    if interference is not None and np.any(interference):
        scaled_map = least_squares(img_scores, interference)
        max_scale = max(haci_cfg.get("max_interference_scale", 1.5), 0.0)
        if not np.isfinite(scaled_map) or scaled_map < 0:
            scaled_map = 0.0
        scaled_map = min(scaled_map, max_scale)
        img_scores = (img_scores - interference * scaled_map).clip(min=0)
        residual_ratio = haci_cfg.get("residual_ratio", 0.0)
        if residual_ratio > 0:
            residual_ratio = max(0.0, min(residual_ratio, 1.0))
            positive_interference = np.maximum(interference, 0.0)
            img_scores = img_scores + positive_interference * residual_ratio

    img_scores_list[vis_token_idx] = {
        "map": img_scores.copy(),
        "layer": current_layer,
        "raw_map": raw_img_scores,
    }

    # prepare raw vision input
    if isinstance(vision_shape[0], tuple):
        cv_img = [cv2.cvtColor(np.array(_), cv2.COLOR_RGB2BGR) for _ in vision_input]
    elif len(vision_shape) == 2:
        cv_img = np.array(vision_input)
        if len(cv_img.shape) == 4 and cv_img.shape[0] == 1:
            cv_img = cv_img[0]
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    else: #video
        cv_img = [cv2.cvtColor(np.array(_), cv2.COLOR_RGB2BGR) for _ in vision_input[0]]

    # prepare top candidates
    candi_scores, candi_ids = logit_list[round_idx][0, this_token_idx].topk(3)
    candi_scores = candi_scores.softmax(0)
    candidates = processor.batch_decode([[_] for _ in candi_ids])
    
    # apply the multimodal_process to obtain TAM
    vis_img, img_map = multimodal_process(cv_img, vision_shape, img_scores, txt_scores, txt_all, candidates, candi_scores, vis_token_idx, \
            save_fn, eval_only=eval_only, vis_width=-1 if eval_only else 500)
    
    if save_fn != '' and vis_token_idx < (len(txt_all) - 1) and isinstance(vis_img, np.ndarray):
        os.makedirs(os.path.dirname(save_fn), exist_ok=True)
        cv2.imwrite(save_fn, vis_img)
    
    return img_map
