#!/usr/bin/env python3
"""
Generate and edit images using Alibaba Cloud DashScope API with Qwen image models.

Supports:
- qwen-image-max (default, high quality generation and editing)

For image editing, provide an input image along with an editing prompt.
"""

import os
import sys
import json
import base64
import argparse
import urllib.request
from pathlib import Path
from typing import Optional


def check_env_file() -> Optional[str]:
    """Check if .env file exists and contains DASHSCOPE_API_KEY."""
    # Look for .env in current directory and parent directories
    current_dir = Path.cwd()
    for parent in [current_dir] + list(current_dir.parents):
        env_file = parent / ".env"
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('DASHSCOPE_API_KEY='):
                        api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                        if api_key:
                            return api_key
    return None


def load_image_as_base64(image_path: str) -> str:
    """Load an image file and return it as a base64 data URL."""
    path = Path(image_path)
    if not path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    # Determine MIME type from extension
    ext = path.suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    mime_type = mime_types.get(ext, 'image/png')

    with open(path, 'rb') as f:
        image_data = f.read()

    base64_data = base64.b64encode(image_data).decode('utf-8')
    return f"data:{mime_type};base64,{base64_data}"


def save_base64_image(base64_data: str, output_path: str) -> None:
    """Save base64 encoded image to file."""
    # Remove data URL prefix if present
    if ',' in base64_data:
        base64_data = base64_data.split(',', 1)[1]

    # Decode and save
    image_data = base64.b64decode(base64_data)
    with open(output_path, 'wb') as f:
        f.write(image_data)


def generate_image(
    prompt: str,
    model: str = "qwen-image-max",
    output_path: str = "generated_image.png",
    api_key: Optional[str] = None,
    input_image: Optional[str] = None
) -> dict:
    """
    Generate or edit an image using DashScope API.

    Args:
        prompt: Text description of the image to generate, or editing instructions
        model: DashScope model ID (default: qwen-image-max)
        output_path: Path to save the generated image
        api_key: DashScope API key (will check .env and DASHSCOPE_API_KEY env var if not provided)
        input_image: Path to an input image for editing (optional)

    Returns:
        dict: Response status information
    """
    try:
        from dashscope import MultiModalConversation
    except ImportError:
        print("Error: 'dashscope' library not found. Install with: pip install dashscope")
        sys.exit(1)

    # Check for API key
    if not api_key:
        api_key = os.environ.get("DASHSCOPE_API_KEY") or check_env_file()

    if not api_key:
        print("Error: DASHSCOPE_API_KEY not found!")
        print("\nPlease create a .env file in your project directory with:")
        print("DASHSCOPE_API_KEY=your-api-key-here")
        print("\nOr set the environment variable:")
        print("export DASHSCOPE_API_KEY=your-api-key-here")
        print("\nGet your API key from: https://dashscope.console.aliyun.com/")
        sys.exit(1)

    # Set API key for DashScope SDK
    os.environ["DASHSCOPE_API_KEY"] = api_key

    # Determine if this is generation or editing
    is_editing = input_image is not None

    if is_editing:
        print(f"Editing image with model: {model}")
        print(f"Input image: {input_image}")
        print(f"Edit prompt: {prompt}")

        # Load input image as base64 data URL
        image_data_url = load_image_as_base64(input_image)

        # Build multimodal message content for image editing
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": prompt},
                    {"image": image_data_url}
                ]
            }
        ]
    else:
        print(f"Generating image with model: {model}")
        print(f"Prompt: {prompt}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"text": prompt}
                ]
            }
        ]

    # Make API request via DashScope SDK
    response = MultiModalConversation.call(
        model=model,
        messages=messages,
        result_format='message',
        stream=False,
        watermark=False,
        prompt_extend=True,
        size='1328*1328'
    )

    # Check for errors
    if response.status_code != 200:
        print(f"API Error ({response.status_code}): {response.code} - {response.message}")
        sys.exit(1)

    # Extract and save image from response
    result = {"status": "success", "model": model}
    image_saved = False

    try:
        content = response.output.choices[0].message.content
        for item in content:
            if "image" in item:
                image_url = item["image"]

                # Check if it's a URL (http/https) or a base64 data URL
                if image_url.startswith("http://") or image_url.startswith("https://"):
                    # Download image from URL
                    print(f"Downloading image from URL...")
                    urllib.request.urlretrieve(image_url, output_path)
                    print(f"Image saved to: {output_path}")
                elif image_url.startswith("data:"):
                    # Save base64 data URL
                    save_base64_image(image_url, output_path)
                    print(f"Image saved to: {output_path}")
                else:
                    # Treat as raw base64
                    save_base64_image(image_url, output_path)
                    print(f"Image saved to: {output_path}")

                image_saved = True
                result["output_path"] = output_path
                break
    except (AttributeError, IndexError, KeyError) as e:
        print(f"Error extracting image from response: {e}")
        print(f"Response: {response}")
        sys.exit(1)

    if not image_saved:
        print("No image found in response")
        try:
            content = response.output.choices[0].message.content
            print(f"Response content: {content}")
        except Exception:
            print(f"Response: {response}")
        sys.exit(1)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate or edit images using DashScope API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with default model (qwen-image-max)
  python generate_image.py "A beautiful sunset over mountains"

  # Specify output path
  python generate_image.py "Abstract art" --output my_image.png

  # Edit an existing image
  python generate_image.py "Make the sky purple" --input photo.jpg --output edited.png

Available models:
  - qwen-image-max (default, high quality, generation + editing)
        """
    )

    parser.add_argument(
        "prompt",
        type=str,
        help="Text description of the image to generate, or editing instructions"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen-image-max",
        help="DashScope model ID (default: qwen-image-max)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="generated_image.png",
        help="Output file path (default: generated_image.png)"
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input image path for editing (enables edit mode)"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="DashScope API key (will check DASHSCOPE_API_KEY env var and .env if not provided)"
    )

    args = parser.parse_args()

    generate_image(
        prompt=args.prompt,
        model=args.model,
        output_path=args.output,
        api_key=args.api_key,
        input_image=args.input
    )


if __name__ == "__main__":
    main()
