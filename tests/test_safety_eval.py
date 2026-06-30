import unittest

from src.evaluation.safety_eval import parse_llamaguard_label


class LlamaGuardParserTest(unittest.TestCase):
    def test_parse_llamaguard_label_safe(self) -> None:
        self.assertEqual(parse_llamaguard_label("safe"), "safe")
        self.assertEqual(parse_llamaguard_label("safe\n"), "safe")

    def test_parse_llamaguard_label_unsafe_with_category(self) -> None:
        self.assertEqual(parse_llamaguard_label("unsafe\nS1"), "unsafe")

    def test_parse_llamaguard_label_error(self) -> None:
        self.assertEqual(parse_llamaguard_label(""), "error")
        self.assertEqual(parse_llamaguard_label("I am not sure"), "error")


if __name__ == "__main__":
    unittest.main()
