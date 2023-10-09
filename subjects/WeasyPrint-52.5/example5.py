
from typing import Union, Tuple
class Gradient:
    def __init__(self, color_stops, repeating):
        assert color_stops
        #: List of (r, g, b, a), list of Dimension
        self.colors = [color for color, position in color_stops]
        self.stop_positions = [position for color, position in color_stops]
        #: bool
        self.repeating = repeating

    def get_intrinsic_size(self, _image_resolution, _font_size):
        # Gradients are not affected by image resolution, parent or font size.
        return None, None

    intrinsic_ratio = None

    def draw(self, context, concrete_width, concrete_height, _image_rendering):
        scale_y, type_, init, stop_positions, stop_colors = self.layout(
            concrete_width, concrete_height, context.user_to_device_distance)
        context.scale(1, scale_y)
        pattern = PATTERN_TYPES[type_](*init)
        for position, color in zip(stop_positions, stop_colors):
            pattern.add_color_stop_rgba(position, *color)
        pattern.set_extend(cairocffi.EXTEND_REPEAT if self.repeating
                           else cairocffi.EXTEND_PAD)
        context.set_source(pattern)
        context.paint()

    def layout(self, width, height, user_to_device_distance):
        """width, height: Gradient box. Top-left is at coordinates (0, 0).
        user_to_device_distance: a (dx, dy) -> (ddx, ddy) function

        Returns (scale_y, type_, init, positions, colors).
        scale_y: float, used for ellipses radial gradients. 1 otherwise.
        positions: list of floats in [0..1].
                   0 at the starting point, 1 at the ending point.
        colors: list of (r, g, b, a)
        type_ is either:
            'solid': init is (r, g, b, a). positions and colors are empty.
            'linear': init is (x0, y0, x1, y1)
                      coordinates of the starting and ending points.
            'radial': init is (cx0, cy0, radius0, cx1, cy1, radius1)
                      coordinates of the starting end ending circles

        """
        raise NotImplementedError


class RadialGradient(Gradient):
    size_type: str
    size: Union[Tuple[str, str], str]
    def __init__(self, color_stops, shape, size, center, repeating):
        Gradient.__init__(self, color_stops, repeating)
        # Center of the ending shape. (origin_x, pos_x, origin_y, pos_y)
        self.center = center
        #: Type of ending shape: 'circle' or 'ellipse'
        self.shape = shape
        # size_type: 'keyword'
        #   size: 'closest-corner', 'farthest-corner',
        #         'closest-side', or 'farthest-side'
        # size_type: 'explicit'
        #   size: (radius_x, radius_y)
        self.size_type, self.size = size

    def layout(self, width, height, user_to_device_distance):
        if len(self.colors) == 1:
            return 1, 'solid', self.colors[0], [], []
        origin_x, center_x, origin_y, center_y = self.center
        center_x = percentage(center_x, width)
        center_y = percentage(center_y, height)
        if origin_x == 'right':
            center_x = width - center_x
        if origin_y == 'bottom':
            center_y = height - center_y

        size_x, size_y = self._resolve_size(width, height, center_x, center_y)
        # http://dev.w3.org/csswg/css-images-3/#degenerate-radials
        if size_x == size_y == 0:
            size_x = size_y = 1e-7
        elif size_x == 0:
            size_x = 1e-7
            size_y = 1e7
        elif size_y == 0:
            size_x = 1e7
            size_y = 1e-7
        scale_y = size_y / size_x

        colors = self.colors
        positions = process_color_stops(size_x, self.stop_positions)
        gradient_line_size = positions[-1] - positions[0]
        if self.repeating and any(
            gradient_line_size * unit < len(positions)
            for unit in (math.hypot(*user_to_device_distance(1, 0)),
                         math.hypot(*user_to_device_distance(0, scale_y)))):
            color = gradient_average_color(colors, positions)
            return 1, 'solid', color, [], []

        if positions[0] < 0:
            # Cairo does not like negative radiuses,
            # shift into the positive realm.
            if self.repeating:
                offset = gradient_line_size * math.ceil(
                    -positions[0] / gradient_line_size)
                positions = [p + offset for p in positions]
            else:
                for i, position in enumerate(positions):
                    if position > 0:
                        # `i` is the first positive stop.
                        # Interpolate with the previous to get the color at 0.
                        assert i > 0
                        color = colors[i]
                        neg_color = colors[i - 1]
                        neg_position = positions[i - 1]
                        assert neg_position < 0
                        intermediate_color = gradient_average_color(
                            [neg_color, neg_color, color, color],
                            [neg_position, 0, 0, position])
                        colors = [intermediate_color] + colors[i:]
                        positions = [0] + positions[i:]
                        break
                else:
                    # All stops are negatives,
                    # everything is "padded" with the last color.
                    return 1, 'solid', self.colors[-1], [], []

        first, last, positions = normalize_stop_postions(positions)
        if last == first:
            last += 100  # Arbitrary non-zero

        circles = (center_x, center_y / scale_y, first,
                   center_x, center_y / scale_y, last)
        return scale_y, 'radial', circles, positions, colors

    def _resolve_size(self, width, height, center_x, center_y):
        if self.size_type == 'explicit':
            size_x, size_y = self.size
            size_x = percentage(size_x, width)
            size_y = percentage(size_y, height)
            return size_x, size_y
        left = abs(center_x)
        right = abs(width - center_x)
        top = abs(center_y)
        bottom = abs(height - center_y)
        pick = min if self.size.startswith('closest') else max
        if self.size.endswith('side'):
            if self.shape == 'circle':
                size_xy = pick(left, right, top, bottom)
                return size_xy, size_xy
            # else: ellipse
            return pick(left, right), pick(top, bottom)
        # else: corner
        if self.shape == 'circle':
            size_xy = pick(math.hypot(left, top), math.hypot(left, bottom),
                           math.hypot(right, top), math.hypot(right, bottom))
            return size_xy, size_xy
        # else: ellipse
        corner_x, corner_y = pick(
            (left, top), (left, bottom), (right, top), (right, bottom),
            key=lambda a: math.hypot(*a))
        return corner_x * math.sqrt(2), corner_y * math.sqrt(2)
