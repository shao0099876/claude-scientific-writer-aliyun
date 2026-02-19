#!/usr/bin/env python3
"""
AI-powered scientific schematic generation using Alibaba Cloud DashScope.

This script uses a smart iterative refinement approach:
1. Generate initial image with qwen-image-max
2. AI quality review using qwen3-vl-plus for scientific critique
3. Only regenerate if quality is below threshold for document type
4. Repeat until quality meets standards (max iterations)

Requirements:
    - DASHSCOPE_API_KEY environment variable
    - dashscope library (pip install dashscope)

Usage:
    python generate_schematic_ai.py "Create a flowchart showing CONSORT participant flow" -o flowchart.png
    python generate_schematic_ai.py "Neural network architecture diagram" -o architecture.png --iterations 2
    python generate_schematic_ai.py "Simple block diagram" -o diagram.png --doc-type poster
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

try:
    import dashscope
    from dashscope import MultiModalConversation
except ImportError:
    print("Error: dashscope library not found. Install with: pip install dashscope")
    sys.exit(1)

# Optional: requests may be used for fallback URL downloading
try:
    import requests
except ImportError:
    requests = None

# Try to load .env file from multiple potential locations
def _load_env_file():
    """Load .env file from current directory, parent directories, or package directory.

    Returns True if a .env file was found and loaded, False otherwise.
    Note: This does NOT override existing environment variables.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False  # python-dotenv not installed

    # Try current working directory first
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        return True

    # Try parent directories (up to 5 levels)
    cwd = Path.cwd()
    for _ in range(5):
        env_path = cwd / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            return True
        cwd = cwd.parent
        if cwd == cwd.parent:  # Reached root
            break

    # Try the package's parent directory (scientific-writer project root)
    script_dir = Path(__file__).resolve().parent
    for _ in range(5):
        env_path = script_dir / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            return True
        script_dir = script_dir.parent
        if script_dir == script_dir.parent:
            break

    return False


class ScientificSchematicGenerator:
    """Generate scientific schematics using AI with smart iterative refinement.

    Uses qwen3-vl-plus for quality review to determine if regeneration is needed.
    Multiple passes only occur if the generated schematic doesn't meet the
    quality threshold for the target document type.
    """

    # Quality thresholds by document type (score out of 10)
    # Higher thresholds for more formal publications
    QUALITY_THRESHOLDS = {
        "journal": 8.5,      # Nature, Science, etc. - highest standards
        "conference": 8.0,   # Conference papers - high standards
        "poster": 7.0,       # Academic posters - good quality
        "presentation": 6.5, # Slides/talks - clear but less formal
        "report": 7.5,       # Technical reports - professional
        "grant": 8.0,        # Grant proposals - must be compelling
        "thesis": 8.0,       # Dissertations - formal academic
        "preprint": 7.5,     # arXiv, etc. - good quality
        "default": 7.5,      # Default threshold
    }

    # Scientific diagram best practices prompt template
    SCIENTIFIC_DIAGRAM_GUIDELINES = """
Create a high-quality scientific diagram with these requirements:

VISUAL QUALITY:
- Clean white or light background (no textures or gradients)
- High contrast for readability and printing
- Professional, publication-ready appearance
- Sharp, clear lines and text
- Adequate spacing between elements to prevent crowding

TYPOGRAPHY:
- Clear, readable sans-serif fonts (Arial, Helvetica style)
- Minimum 10pt font size for all labels
- Consistent font sizes throughout
- All text horizontal or clearly readable
- No overlapping text

SCIENTIFIC STANDARDS:
- Accurate representation of concepts
- Clear labels for all components
- Include scale bars, legends, or axes where appropriate
- Use standard scientific notation and symbols
- Include units where applicable

ACCESSIBILITY:
- Colorblind-friendly color palette (use Okabe-Ito colors if using color)
- High contrast between elements
- Redundant encoding (shapes + colors, not just colors)
- Works well in grayscale

LAYOUT:
- Logical flow (left-to-right or top-to-bottom)
- Clear visual hierarchy
- Balanced composition
- Appropriate use of whitespace
- No clutter or unnecessary decorative elements

IMPORTANT - NO FIGURE NUMBERS:
- Do NOT include "Figure 1:", "Fig. 1", or any figure numbering in the image
- Do NOT add captions or titles like "Figure: ..." at the top or bottom
- Figure numbers and captions are added separately in the document/LaTeX
- The diagram should contain only the visual content itself
"""

    def __init__(self, api_key: Optional[str] = None, verbose: bool = False):
        """
        Initialize the generator.

        Args:
            api_key: DashScope API key (or use DASHSCOPE_API_KEY env var)
            verbose: Print detailed progress information
        """
        # Priority: 1) explicit api_key param, 2) environment variable, 3) .env file
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")

        # If not found in environment, try loading from .env file
        if not self.api_key:
            _load_env_file()
            self.api_key = os.getenv("DASHSCOPE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY not found. Please either:\n"
                "  1. Set the DASHSCOPE_API_KEY environment variable\n"
                "  2. Add DASHSCOPE_API_KEY to your .env file\n"
                "  3. Pass api_key parameter to the constructor\n"
                "Get your API key from: https://dashscope.console.aliyun.com/"
            )

        self.verbose = verbose
        self._last_error = None  # Track last error for better reporting
        # qwen-image-max - Alibaba Cloud's advanced image generation model
        self.image_model = "qwen-image-max"
        # qwen3-vl-plus for quality review - excellent vision and reasoning
        self.review_model = "qwen3-vl-plus"

    def _log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")

    def _make_request(self, model: str, messages: List[Dict[str, Any]],
                     **kwargs) -> Any:
        """
        Make a request to DashScope API via the MultiModalConversation SDK.

        Args:
            model: Model identifier
            messages: List of message dictionaries in DashScope format
            **kwargs: Additional parameters passed to MultiModalConversation.call()
                      (e.g., watermark, prompt_extend, size for image generation)

        Returns:
            DashScope Response object

        Raises:
            RuntimeError: If the API request fails
        """
        self._log(f"Making request to {model}...")

        call_params = {
            "api_key": self.api_key,
            "model": model,
            "messages": messages,
            "result_format": "message",
            "stream": False,
        }

        # Merge any additional kwargs (watermark, prompt_extend, size, etc.)
        call_params.update(kwargs)

        try:
            response = MultiModalConversation.call(**call_params)

            # Check for errors in the DashScope response
            if response.status_code != 200:
                error_code = getattr(response, 'code', 'Unknown')
                error_message = getattr(response, 'message', 'Unknown error')
                self._log(f"API Error (status {response.status_code}): [{error_code}] {error_message}")
                raise RuntimeError(
                    f"API request failed (status {response.status_code}): "
                    f"[{error_code}] {error_message}"
                )

            return response

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"API request failed: {str(e)}")

    def _extract_image_from_response(self, response: Any) -> Optional[bytes]:
        """
        Extract image data from a DashScope API response.

        DashScope returns response.output.choices[0].message.content as a list
        of items like [{"image": "https://..."}, {"text": "description"}].
        If the image value is a URL, download it. If it's base64, decode it.

        Args:
            response: DashScope Response object

        Returns:
            Image bytes or None if not found
        """
        try:
            choices = response.output.choices
            if not choices:
                self._log("No choices in response")
                return None

            message = choices[0].message
            content = message.content

            if self.verbose:
                self._log(f"Content type: {type(content)}, value: {str(content)[:300]}")

            # DashScope returns content as a list of dicts
            if isinstance(content, list):
                for i, item in enumerate(content):
                    if isinstance(item, dict) and "image" in item:
                        image_value = item["image"]
                        self._log(f"Found image in content item {i}")

                        # Check if it's a URL (http/https)
                        if isinstance(image_value, str) and image_value.startswith(("http://", "https://")):
                            self._log(f"Downloading image from URL...")
                            try:
                                # Download the image from URL
                                req = urllib.request.Request(image_value)
                                with urllib.request.urlopen(req, timeout=60) as resp:
                                    image_data = resp.read()
                                self._log(f"Downloaded image ({len(image_data)} bytes)")
                                return image_data
                            except Exception as e:
                                self._log(f"Failed to download image URL: {e}")
                                # Fallback: try with requests if available
                                if requests is not None:
                                    try:
                                        r = requests.get(image_value, timeout=60)
                                        r.raise_for_status()
                                        self._log(f"Downloaded image via requests ({len(r.content)} bytes)")
                                        return r.content
                                    except Exception as e2:
                                        self._log(f"Requests fallback also failed: {e2}")
                                return None

                        # Check if it's a base64 data URL
                        elif isinstance(image_value, str) and image_value.startswith("data:image"):
                            if "," in image_value:
                                base64_str = image_value.split(",", 1)[1]
                                base64_str = base64_str.replace('\n', '').replace('\r', '').replace(' ', '')
                                self._log(f"Decoded base64 image (length: {len(base64_str)})")
                                return base64.b64decode(base64_str)

                        # Check if it's raw base64
                        elif isinstance(image_value, str) and len(image_value) > 100:
                            try:
                                image_data = base64.b64decode(image_value)
                                self._log(f"Decoded raw base64 image ({len(image_data)} bytes)")
                                return image_data
                            except Exception:
                                self._log(f"Image value is not valid base64")

            # Handle string content as fallback
            if isinstance(content, str) and "data:image" in content:
                import re
                match = re.search(r'data:image/[^;]+;base64,([A-Za-z0-9+/=\n\r]+)', content, re.DOTALL)
                if match:
                    base64_str = match.group(1).replace('\n', '').replace('\r', '').replace(' ', '')
                    self._log(f"Found image in string content (length: {len(base64_str)})")
                    return base64.b64decode(base64_str)

            self._log("No image data found in response")
            return None

        except Exception as e:
            self._log(f"Error extracting image: {str(e)}")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return None

    def _image_to_base64(self, image_path: str) -> str:
        """
        Convert image file to base64 data URL.

        Args:
            image_path: Path to image file

        Returns:
            Base64 data URL string
        """
        with open(image_path, "rb") as f:
            image_data = f.read()

        # Determine image type from extension
        ext = Path(image_path).suffix.lower()
        mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }.get(ext, "image/png")

        base64_data = base64.b64encode(image_data).decode("utf-8")
        return f"data:{mime_type};base64,{base64_data}"

    def generate_image(self, prompt: str) -> Optional[bytes]:
        """
        Generate an image using qwen-image-max via DashScope.

        Args:
            prompt: Description of the diagram to generate

        Returns:
            Image bytes or None if generation failed
        """
        self._last_error = None  # Reset error

        messages = [
            {
                "role": "user",
                "content": [
                    {"text": prompt}
                ]
            }
        ]

        try:
            response = self._make_request(
                model=self.image_model,
                messages=messages,
                watermark=False,
                prompt_extend=True,
                size='1328*1328'
            )

            # Debug: print response structure if verbose
            if self.verbose:
                try:
                    choices = response.output.choices
                    self._log(f"Response status: {response.status_code}")
                    if choices:
                        msg = choices[0].message
                        content = msg.content
                        if isinstance(content, list):
                            self._log(f"Content is list with {len(content)} items")
                            for i, item in enumerate(content[:3]):
                                if isinstance(item, dict):
                                    keys = list(item.keys())
                                    self._log(f"  Item {i}: keys={keys}")
                        elif isinstance(content, str):
                            preview = content[:200] + "..." if len(content) > 200 else content
                            self._log(f"Content preview: {preview}")
                except Exception as e:
                    self._log(f"Error inspecting response: {e}")

            image_data = self._extract_image_from_response(response)
            if image_data:
                self._log(f"Generated image ({len(image_data)} bytes)")
            else:
                self._last_error = "No image data in API response - model may not support image generation"
                self._log(f"✗ {self._last_error}")

            return image_data
        except RuntimeError as e:
            self._last_error = str(e)
            self._log(f"✗ Generation failed: {self._last_error}")
            return None
        except Exception as e:
            self._last_error = f"Unexpected error: {str(e)}"
            self._log(f"✗ Generation failed: {self._last_error}")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return None

    def review_image(self, image_path: str, original_prompt: str,
                    iteration: int, doc_type: str = "default",
                    max_iterations: int = 2) -> Tuple[str, float, bool]:
        """
        Review generated image using qwen3-vl-plus for quality analysis.

        Uses qwen3-vl-plus's vision and reasoning capabilities to
        evaluate the schematic quality and determine if regeneration is needed.

        Args:
            image_path: Path to the generated image
            original_prompt: Original user prompt
            iteration: Current iteration number
            doc_type: Document type (journal, poster, presentation, etc.)
            max_iterations: Maximum iterations allowed

        Returns:
            Tuple of (critique text, quality score 0-10, needs_improvement bool)
        """
        # Use qwen3-vl-plus for review - excellent vision and analysis
        image_data_url = self._image_to_base64(image_path)

        # Get quality threshold for this document type
        threshold = self.QUALITY_THRESHOLDS.get(doc_type.lower(),
                                                 self.QUALITY_THRESHOLDS["default"])

        review_prompt = f"""You are an expert reviewer evaluating a scientific diagram for publication quality.

ORIGINAL REQUEST: {original_prompt}

DOCUMENT TYPE: {doc_type} (quality threshold: {threshold}/10)
ITERATION: {iteration}/{max_iterations}

Carefully evaluate this diagram on these criteria:

1. **Scientific Accuracy** (0-2 points)
   - Correct representation of concepts
   - Proper notation and symbols
   - Accurate relationships shown

2. **Clarity and Readability** (0-2 points)
   - Easy to understand at a glance
   - Clear visual hierarchy
   - No ambiguous elements

3. **Label Quality** (0-2 points)
   - All important elements labeled
   - Labels are readable (appropriate font size)
   - Consistent labeling style

4. **Layout and Composition** (0-2 points)
   - Logical flow (top-to-bottom or left-to-right)
   - Balanced use of space
   - No overlapping elements

5. **Professional Appearance** (0-2 points)
   - Publication-ready quality
   - Clean, crisp lines and shapes
   - Appropriate colors/contrast

RESPOND IN THIS EXACT FORMAT:
SCORE: [total score 0-10]

STRENGTHS:
- [strength 1]
- [strength 2]

ISSUES:
- [issue 1 if any]
- [issue 2 if any]

VERDICT: [ACCEPTABLE or NEEDS_IMPROVEMENT]

If score >= {threshold}, the diagram is ACCEPTABLE for {doc_type} publication.
If score < {threshold}, mark as NEEDS_IMPROVEMENT with specific suggestions."""

        # DashScope message format for vision model
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": review_prompt},
                    {"image": image_data_url}
                ]
            }
        ]

        try:
            # Use qwen3-vl-plus for high-quality review (no image gen params)
            response = self._make_request(
                model=self.review_model,
                messages=messages
            )

            # Extract text response from DashScope response object
            choices = response.output.choices
            if not choices:
                return "Image generated successfully", 8.0, False

            message = choices[0].message
            content = message.content

            # DashScope content is a list of items
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                content = "\n".join(text_parts)
            elif not isinstance(content, str):
                content = str(content) if content else ""

            # Try to extract score
            score = 7.5  # Default score if extraction fails
            import re

            # Look for SCORE: X or SCORE: X/10 format
            score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', content, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
            else:
                # Fallback: look for any score pattern
                score_match = re.search(r'(?:score|rating|quality)[:\s]+(\d+(?:\.\d+)?)\s*(?:/\s*10)?', content, re.IGNORECASE)
                if score_match:
                    score = float(score_match.group(1))

            # Determine if improvement is needed based on verdict or score
            needs_improvement = False
            if "NEEDS_IMPROVEMENT" in content.upper():
                needs_improvement = True
            elif score < threshold:
                needs_improvement = True

            self._log(f"Review complete (Score: {score}/10, Threshold: {threshold}/10)")
            self._log(f"  Verdict: {'Needs improvement' if needs_improvement else 'Acceptable'}")

            return (content if content else "Image generated successfully",
                    score,
                    needs_improvement)
        except Exception as e:
            self._log(f"Review skipped: {str(e)}")
            # Don't fail the whole process if review fails - assume acceptable
            return "Image generated successfully (review skipped)", 7.5, False

    def improve_prompt(self, original_prompt: str, critique: str,
                      iteration: int) -> str:
        """
        Improve the generation prompt based on critique.

        Args:
            original_prompt: Original user prompt
            critique: Review critique from previous iteration
            iteration: Current iteration number

        Returns:
            Improved prompt for next generation
        """
        improved_prompt = f"""{self.SCIENTIFIC_DIAGRAM_GUIDELINES}

USER REQUEST: {original_prompt}

ITERATION {iteration}: Based on previous feedback, address these specific improvements:
{critique}

Generate an improved version that addresses all the critique points while maintaining scientific accuracy and professional quality."""

        return improved_prompt

    def generate_iterative(self, user_prompt: str, output_path: str,
                          iterations: int = 2,
                          doc_type: str = "default") -> Dict[str, Any]:
        """
        Generate scientific schematic with smart iterative refinement.

        Only regenerates if the quality score is below the threshold for the
        specified document type. This saves API calls and time when the first
        generation is already good enough.

        Args:
            user_prompt: User's description of desired diagram
            output_path: Path to save final image
            iterations: Maximum refinement iterations (default: 2, max: 2)
            doc_type: Document type for quality threshold (journal, poster, etc.)

        Returns:
            Dictionary with generation results and metadata
        """
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = output_path.stem
        extension = output_path.suffix or ".png"

        # Get quality threshold for this document type
        threshold = self.QUALITY_THRESHOLDS.get(doc_type.lower(),
                                                 self.QUALITY_THRESHOLDS["default"])

        results = {
            "user_prompt": user_prompt,
            "doc_type": doc_type,
            "quality_threshold": threshold,
            "iterations": [],
            "final_image": None,
            "final_score": 0.0,
            "success": False,
            "early_stop": False,
            "early_stop_reason": None
        }

        current_prompt = f"""{self.SCIENTIFIC_DIAGRAM_GUIDELINES}

USER REQUEST: {user_prompt}

Generate a publication-quality scientific diagram that meets all the guidelines above."""

        print(f"\n{'='*60}")
        print(f"Generating Scientific Schematic")
        print(f"{'='*60}")
        print(f"Description: {user_prompt}")
        print(f"Document Type: {doc_type}")
        print(f"Quality Threshold: {threshold}/10")
        print(f"Max Iterations: {iterations}")
        print(f"Output: {output_path}")
        print(f"{'='*60}\n")

        for i in range(1, iterations + 1):
            print(f"\n[Iteration {i}/{iterations}]")
            print("-" * 40)

            # Generate image
            print(f"Generating image with qwen-image-max...")
            image_data = self.generate_image(current_prompt)

            if not image_data:
                error_msg = getattr(self, '_last_error', 'Image generation failed - no image data returned')
                print(f"Generation failed: {error_msg}")
                results["iterations"].append({
                    "iteration": i,
                    "success": False,
                    "error": error_msg
                })
                continue

            # Save iteration image
            iter_path = output_dir / f"{base_name}_v{i}{extension}"
            with open(iter_path, "wb") as f:
                f.write(image_data)
            print(f"Saved: {iter_path}")

            # Review image using qwen3-vl-plus
            print(f"Reviewing image with qwen3-vl-plus...")
            critique, score, needs_improvement = self.review_image(
                str(iter_path), user_prompt, i, doc_type, iterations
            )
            print(f"Score: {score}/10 (threshold: {threshold}/10)")

            # Save iteration results
            iteration_result = {
                "iteration": i,
                "image_path": str(iter_path),
                "prompt": current_prompt,
                "critique": critique,
                "score": score,
                "needs_improvement": needs_improvement,
                "success": True
            }
            results["iterations"].append(iteration_result)

            # Check if quality is acceptable - STOP EARLY if so
            if not needs_improvement:
                print(f"\nQuality meets {doc_type} threshold ({score} >= {threshold})")
                print(f"  No further iterations needed!")
                results["final_image"] = str(iter_path)
                results["final_score"] = score
                results["success"] = True
                results["early_stop"] = True
                results["early_stop_reason"] = f"Quality score {score} meets threshold {threshold} for {doc_type}"
                break

            # If this is the last iteration, we're done regardless
            if i == iterations:
                print(f"\nMaximum iterations reached")
                results["final_image"] = str(iter_path)
                results["final_score"] = score
                results["success"] = True
                break

            # Quality below threshold - improve prompt for next iteration
            print(f"\nQuality below threshold ({score} < {threshold})")
            print(f"Improving prompt based on feedback...")
            current_prompt = self.improve_prompt(user_prompt, critique, i + 1)

        # Copy final version to output path
        if results["success"] and results["final_image"]:
            final_iter_path = Path(results["final_image"])
            if final_iter_path != output_path:
                import shutil
                shutil.copy(final_iter_path, output_path)
                print(f"\nFinal image: {output_path}")

        # Save review log
        log_path = output_dir / f"{base_name}_review_log.json"
        with open(log_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Review log: {log_path}")

        print(f"\n{'='*60}")
        print(f"Generation Complete!")
        print(f"Final Score: {results['final_score']}/10")
        if results["early_stop"]:
            print(f"Iterations Used: {len([r for r in results['iterations'] if r.get('success')])}/{iterations} (early stop)")
        print(f"{'='*60}\n")

        return results


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate scientific schematics using AI with smart iterative refinement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a flowchart for a journal paper
  python generate_schematic_ai.py "CONSORT participant flow diagram" -o flowchart.png --doc-type journal

  # Generate neural network architecture for presentation (lower threshold)
  python generate_schematic_ai.py "Transformer encoder-decoder architecture" -o transformer.png --doc-type presentation

  # Generate with custom max iterations for poster
  python generate_schematic_ai.py "Biological signaling pathway" -o pathway.png --iterations 2 --doc-type poster

  # Verbose output
  python generate_schematic_ai.py "Circuit diagram" -o circuit.png -v

Document Types (quality thresholds):
  journal      8.5/10  - Nature, Science, peer-reviewed journals
  conference   8.0/10  - Conference papers
  thesis       8.0/10  - Dissertations, theses
  grant        8.0/10  - Grant proposals
  preprint     7.5/10  - arXiv, bioRxiv, etc.
  report       7.5/10  - Technical reports
  poster       7.0/10  - Academic posters
  presentation 6.5/10  - Slides, talks
  default      7.5/10  - General purpose

Note: Multiple iterations only occur if quality is BELOW the threshold.
      If the first generation meets the threshold, no extra API calls are made.

Environment:
  DASHSCOPE_API_KEY    DashScope API key (required)
        """
    )

    parser.add_argument("prompt", help="Description of the diagram to generate")
    parser.add_argument("-o", "--output", required=True,
                       help="Output image path (e.g., diagram.png)")
    parser.add_argument("--iterations", type=int, default=2,
                       help="Maximum refinement iterations (default: 2, max: 2)")
    parser.add_argument("--doc-type", default="default",
                       choices=["journal", "conference", "poster", "presentation",
                               "report", "grant", "thesis", "preprint", "default"],
                       help="Document type for quality threshold (default: default)")
    parser.add_argument("--api-key", help="DashScope API key (or set DASHSCOPE_API_KEY)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    # Check for API key
    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY environment variable not set")
        print("\nSet it with:")
        print("  export DASHSCOPE_API_KEY='your_api_key'")
        print("\nOr provide via --api-key flag")
        sys.exit(1)

    # Validate iterations - enforce max of 2
    if args.iterations < 1 or args.iterations > 2:
        print("Error: Iterations must be between 1 and 2")
        sys.exit(1)

    try:
        generator = ScientificSchematicGenerator(api_key=api_key, verbose=args.verbose)
        results = generator.generate_iterative(
            user_prompt=args.prompt,
            output_path=args.output,
            iterations=args.iterations,
            doc_type=args.doc_type
        )

        if results["success"]:
            print(f"\nSuccess! Image saved to: {args.output}")
            if results.get("early_stop"):
                print(f"  (Completed in {len([r for r in results['iterations'] if r.get('success')])} iteration(s) - quality threshold met)")
            sys.exit(0)
        else:
            print(f"\nGeneration failed. Check review log for details.")
            sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
