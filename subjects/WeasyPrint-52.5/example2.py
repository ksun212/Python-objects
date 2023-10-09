from typing import Union


class OrientedBox:
    @property
    def sugar(self):
        return self.padding_plus_border + self.margin_a + self.margin_b

    @property
    def outer(self):
        return self.sugar + self.inner

    @property
    def outer_min_content_size(self):
        return self.sugar + (
            self.min_content_size if self.inner == 'auto' else self.inner)

    @property
    def outer_max_content_size(self):
        return self.sugar + (
            self.max_content_size if self.inner == 'auto' else self.inner)

    def shrink_to_fit(self, available):
        self.inner = min(
            max(self.min_content_size, available), self.max_content_size)


class VerticalBox(OrientedBox):
    margin_a: Union[str, int, float]
    margin_b: Union[str, int, float]
    inner: Union[str, int, float]
    padding_plus_border: int
    def __init__(self, context, box):
        self.context = context
        self.box = box
        # Inner dimension: that of the content area, as opposed to the
        # outer dimension: that of the margin area.
        self.inner = box.height
        self.margin_a = box.margin_top
        self.margin_b = box.margin_bottom
        self.padding_plus_border = (
            box.padding_top + box.padding_bottom +
            box.border_top_width + box.border_bottom_width)

    def restore_box_attributes(self):
        box = self.box
        box.height = self.inner
        box.margin_top = self.margin_a
        box.margin_bottom = self.margin_b

    # TODO: Define what are the min-content and max-content heights
    @property
    def min_content_size(self):
        return 0

    @property
    def max_content_size(self):
        return 1e6

def page_width_or_height(box: VerticalBox, containing_block_size: int) -> None:
    """Take a :class:`OrientedBox` object and set either width, margin-left
    and margin-right; or height, margin-top and margin-bottom.

    "The width and horizontal margins of the page box are then calculated
     exactly as for a non-replaced block element in normal flow. The height
     and vertical margins of the page box are calculated analogously (instead
     of using the block height formulas). In both cases if the values are
     over-constrained, instead of ignoring any margins, the containing block
     is resized to coincide with the margin edges of the page box."

    http://dev.w3.org/csswg/css3-page/#page-box-page-rule
    http://www.w3.org/TR/CSS21/visudet.html#blockwidth

    """
    remaining = containing_block_size - box.padding_plus_border
    if box.inner == 'auto':
        if box.margin_a == 'auto':
            box.margin_a = 0
        if box.margin_b == 'auto':
            box.margin_b = 0    
        box.inner = remaining - box.margin_a - box.margin_b
    elif box.margin_a == box.margin_b == 'auto':
        box.margin_a = box.margin_b = (remaining - box.inner) / 2
    elif box.margin_a == 'auto':
        box.margin_a = remaining - box.inner - box.margin_b
    elif box.margin_b == 'auto':
        box.margin_b = remaining - box.inner - box.margin_a
    box.restore_box_attributes()