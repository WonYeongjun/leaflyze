from PIL import Image


def combine_images_vertically(image_paths, output_path):
    images = [Image.open(image) for image in image_paths]

    # Calculate the total height and the maximum width
    total_height = sum(image.height for image in images)
    max_width = max(image.width for image in images)

    # Create a new blank image with the calculated dimensions
    combined_image = Image.new("RGB", (max_width, total_height))

    # Paste each image into the combined image
    y_offset = 0
    for image in images:
        combined_image.paste(image, (0, y_offset))
        y_offset += image.height

    # Save the combined image
    combined_image.save(output_path)


# Example usage
image_paths = [
    "./output/back/black_back_result.png",
    "./output/back/white_back_result.png",
    "./output/back/purple_back_result.png",
]
# image_paths = [
#     "./back/black_back_result.png",
#     "./back/white_back_result.png",
#     "./back/purple_back_result.png",
# ]
output_path = "./output/back/all_result.png"
combine_images_vertically(image_paths, output_path)
