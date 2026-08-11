"""Word Error Rate (WER) calculation for OCR evaluation."""

from __future__ import annotations

from collections.abc import Sequence


class WERCalculator:
    """Calculate Word Error Rate for OCR evaluation.

    WER measures the word-level accuracy of OCR output by comparing
    the recognized text against ground truth. It counts insertions,
    deletions, and substitutions required to transform the OCR output
    into the ground truth at the word level.
    """

    def calculate(self, ground_truth: str, hypothesis: str) -> float:
        """Calculate WER between ground truth and hypothesis.

        Args:
            ground_truth: The correct reference text.
            hypothesis: The OCR output to evaluate.

        Returns:
            WER as a float between 0.0 and 1.0, where 0.0 is perfect match.

        Examples:
            >>> calc = WERCalculator()
            >>> calc.calculate("hello world", "hello world")
            0.0
            >>> calc.calculate("hello world", "helo world")
            0.5  # 1 substitution out of 2 words
        """
        # Split into words
        truth_words = ground_truth.split()
        hyp_words = hypothesis.split()

        if not truth_words:
            if not hyp_words:
                return 0.0
            return 1.0

        # Calculate edit distance on word level
        distance = self._levenshtein_distance(truth_words, hyp_words)

        # WER = edit distance / number of words in ground truth
        return distance / len(truth_words)

    def _levenshtein_distance(self, s1: Sequence[str], s2: Sequence[str]) -> int:
        """Calculate Levenshtein distance between two sequences.

        Args:
            s1: First sequence (words).
            s2: Second sequence (words).

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
        """Calculate WER for multiple text pairs.

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

        wer_values = [
            self.calculate(gt, hyp)
            for gt, hyp in zip(ground_truths, hypotheses)
        ]

        mean_wer = sum(wer_values) / len(wer_values)
        min_wer = min(wer_values)
        max_wer = max(wer_values)

        # Calculate standard deviation
        if len(wer_values) > 1:
            variance = sum((x - mean_wer) ** 2 for x in wer_values) / len(wer_values)
            std_wer = variance ** 0.5
        else:
            std_wer = 0.0

        return {
            "mean": mean_wer,
            "min": min_wer,
            "max": max_wer,
            "std": std_wer,
        }
