"""Character Error Rate (CER) calculation for OCR evaluation."""

from __future__ import annotations

from collections.abc import Sequence


class CERCalculator:
    """Calculate Character Error Rate for OCR evaluation.

    CER measures the character-level accuracy of OCR output by comparing
    the recognized text against ground truth. It counts insertions,
    deletions, and substitutions required to transform the OCR output
    into the ground truth.
    """

    def calculate(self, ground_truth: str, hypothesis: str) -> float:
        """Calculate CER between ground truth and hypothesis.

        Args:
            ground_truth: The correct reference text.
            hypothesis: The OCR output to evaluate.

        Returns:
            CER as a float between 0.0 and 1.0, where 0.0 is perfect match.

        Examples:
            >>> calc = CERCalculator()
            >>> calc.calculate("hello world", "hello world")
            0.0
            >>> calc.calculate("hello world", "helo world")
            0.1  # 1 substitution out of 10 characters
        """
        if not ground_truth:
            if not hypothesis:
                return 0.0
            return 1.0

        # Calculate edit distance
        distance = self._levenshtein_distance(ground_truth, hypothesis)

        # CER = edit distance / length of ground truth
        return distance / len(ground_truth)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings.

        Args:
            s1: First string.
            s2: Second string.

        Returns:
            Edit distance (number of insertions, deletions, substitutions).
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        # s1 is now longer than s2
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)

        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def calculate_batch(
        self,
        ground_truths: Sequence[str],
        hypotheses: Sequence[str],
    ) -> dict[str, float]:
        """Calculate CER for multiple text pairs.

        Args:
            ground_truths: Sequence of ground truth texts.
            hypotheses: Sequence of hypothesis texts.

        Returns:
            Dictionary with statistics: mean, min, max, std.

        Raises:
            ValueError: If sequences have different lengths.
        """
        if len(ground_truths) != len(hypotheses):
            raise ValueError("Ground truths and hypotheses must have same length")

        if not ground_truths:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

        cer_values = [
            self.calculate(gt, hyp)
            for gt, hyp in zip(ground_truths, hypotheses)
        ]

        mean_cer = sum(cer_values) / len(cer_values)
        min_cer = min(cer_values)
        max_cer = max(cer_values)

        # Calculate standard deviation
        if len(cer_values) > 1:
            variance = sum((x - mean_cer) ** 2 for x in cer_values) / len(cer_values)
            std_cer = variance ** 0.5
        else:
            std_cer = 0.0

        return {
            "mean": mean_cer,
            "min": min_cer,
            "max": max_cer,
            "std": std_cer,
        }
