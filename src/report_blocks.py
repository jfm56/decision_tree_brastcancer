def render_multiline_block(ax, x, y, lines, font_sizes, line_spacings, weights=None, bullet_indent=0.03):
    """
    Helper to render a block of text with variable font sizes, weights, and spacing.
    lines: list of strings
    font_sizes: list of font sizes (same length as lines)
    line_spacings: list of y decrements after each line
    weights: list of font weights ("normal", "bold"), optional
    bullet_indent: how far to indent bullets
    """
    for i, line in enumerate(lines):
        kw = {}
        if weights and weights[i]:
            kw['fontweight'] = weights[i]
        indent = x
        if line.strip().startswith('•') or line.strip().startswith('- '):
            indent += bullet_indent
        ax.text(indent, y, line, fontsize=font_sizes[i], wrap=True, **kw)
        y -= line_spacings[i]
    return y
