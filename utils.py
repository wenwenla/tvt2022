from PIL import ImageDraw


def _draw_with_color_and_width(image, position, transform, color, width):
    p = transform @ position
    draw = ImageDraw.Draw(image)
    draw.ellipse(((p[0] - width / 2, p[1] - width / 2), (p[0] + width / 2, p[1] + width / 2)), fill=color)


def draw_target_area(image, position, transform, r):
    WIDTH = r * 2 * transform[0, 0]
    _draw_with_color_and_width(image, position, transform, 'yellow', WIDTH)


def draw_uav(image, position, transform):
    WIDTH = 3
    _draw_with_color_and_width(image, position, transform, 'red', WIDTH)


def draw_obstacle(image, position, transform, collision, r):
    WIDTH = r * 2 * transform[0, 0]
    _draw_with_color_and_width(image, position, transform, 'grey' if not collision else 'red', WIDTH)


def draw_tr(image, tr, transform):
    WIDTH = 1
    for u in tr:
        for p in u:
            _draw_with_color_and_width(image, p, transform, 'black', WIDTH)


def draw_grid(image, row, col, cell_size):
    height = row * cell_size
    width = col * cell_size
    draw = ImageDraw.Draw(image)
    for r in range(row + 1):
        draw.line(((r * cell_size, 0), (r * cell_size, width)), fill='black')
    for c in range(col + 1):
        draw.line(((0, c * cell_size), (height, c * cell_size)), fill='black')


def draw_ball(image, row, col, cell_size, color='red'):
    pos_x = row * cell_size + cell_size / 2
    pos_y = col * cell_size + cell_size / 2
    draw = ImageDraw.Draw(image)
    draw.ellipse(((pos_x - cell_size / 2, pos_y - cell_size / 2), (pos_x + cell_size / 2, pos_y + cell_size / 2)), fill=color, outline=color)