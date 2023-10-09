class Box:
    """Abstract base class for all boxes."""
    # Definitions for the rules generating anonymous table boxes
    # http://www.w3.org/TR/CSS21/tables.html#anonymous-boxes
    proper_table_child = False
    internal_table_or_caption = False
    tabular_container = False

    # Keep track of removed collapsing spaces for wrap opportunities.
    leading_collapsible_space = False
    trailing_collapsible_space = False

    # Default, may be overriden on instances.
    is_table_wrapper = False
    is_flex_item = False
    is_for_root_element = False
    is_column = False

    # Other properties
    transformation_matrix = None
    bookmark_label = None
    string_set = None
    padding_top: int
    margin_top: int
    position_y: int
    border_top_width: int
    # Default, overriden on some subclasses
    def all_children(self):
        return ()

    def __init__(self, element_tag, style, element):
        self.element_tag = element_tag
        self.element = element
        self.style = style
        self.remove_decoration_sides = set()

    def __repr__(self):
        return '<%s %s>' % (type(self).__name__, self.element_tag)

    @classmethod
    def anonymous_from(cls, parent, *args, **kwargs):
        """Return an anonymous box that inherits from ``parent``."""
        style = computed_from_cascaded(
            cascaded={}, parent_style=parent.style, element=None)
        return cls(parent.element_tag, style, parent.element, *args, **kwargs)

    def copy(self):
        """Return shallow copy of the box."""
        cls = type(self)
        # Create a new instance without calling __init__: parameters are
        # different depending on the class.
        new_box = cls.__new__(cls)
        # Copy attributes
        new_box.__dict__.update(self.__dict__)
        return new_box

    def deepcopy(self):
        """Return a copy of the box with recursive copies of its children."""
        return self.copy()

    def translate(self, dx=0, dy=0, ignore_floats=False):
        """Change the box’s position.

        Also update the children’s positions accordingly.

        """
        # Overridden in ParentBox to also translate children, if any.
        if dx == 0 and dy == 0:
            return
        self.position_x += dx
        self.position_y += dy
        for child in self.all_children():
            if not (ignore_floats and child.is_floated()):
                child.translate(dx, dy, ignore_floats)

    # Heights and widths

    def padding_width(self):
        """Width of the padding box."""
        return self.width + self.padding_left + self.padding_right

    def padding_height(self):
        """Height of the padding box."""
        return self.height + self.padding_top + self.padding_bottom

    def border_width(self):
        """Width of the border box."""
        return self.padding_width() + self.border_left_width + \
            self.border_right_width

    def border_height(self):
        """Height of the border box."""
        return self.padding_height() + self.border_top_width + \
            self.border_bottom_width

    def margin_width(self):
        """Width of the margin box (aka. outer box)."""
        return self.border_width() + self.margin_left + self.margin_right

    def margin_height(self):
        """Height of the margin box (aka. outer box)."""
        return self.border_height() + self.margin_top + self.margin_bottom

    # Corners positions

    def content_box_x(self) -> int:
        """Absolute horizontal position of the content box."""
        return self.position_x + self.margin_left + self.padding_left + \
            self.border_left_width

    def content_box_y(self) -> int:
        """Absolute vertical position of the content box."""
        return self.position_y + self.margin_top + self.padding_top + \
            self.border_top_width