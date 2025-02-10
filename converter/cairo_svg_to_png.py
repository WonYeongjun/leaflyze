import cairo
from svgpathtools import svg2paths

# SVG 파일 경로
svg_file = "C:/Users/UserK/Desktop/new/glass_white.svg"

# SVG 파일에서 경로 추출
paths, attributes = svg2paths(svg_file)

# Cairo surface 설정
width, height = 800, 600  # 크기 설정
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
context = cairo.Context(surface)

# 배경 흰색으로 설정
context.set_source_rgb(1, 1, 1)  # 흰색
context.paint()

# 경로 그리기
context.set_source_rgb(0, 0, 0)  # 검은색

for path in paths:
    for segment in path:
        # 각 경로의 종류에 맞춰 그리기
        if segment.__class__.__name__ == "Line":
            context.move_to(segment.start.real, segment.start.imag)
            context.line_to(segment.end.real, segment.end.imag)
        elif segment.__class__.__name__ == "CubicBezier":
            context.move_to(segment.start.real, segment.start.imag)
            context.curve_to(
                segment.control1.real,
                segment.control1.imag,
                segment.control2.real,
                segment.control2.imag,
                segment.end.real,
                segment.end.imag,
            )
        context.stroke()

# 이미지 저장
surface.write_to_png("C:/Users/UserK/Desktop/new/glass_white.png")
