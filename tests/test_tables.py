"""Reading the generated tables out of the report, and drawing them."""

from pathlib import Path
import tempfile
import unittest

import tables

REPORT = f"""# Kept above the marker

| ignored | table |
|---|---|
| above | the marker |

{tables.MARKER}

## Sensor noise

| sigma_p | A caught | B caught |
|---|---|---|
| 0 mm | 85 / 100 | 85 / 100 |
| 2 mm | 62 / 100 | 87 / 100 |

## Every metric at each close latency

### 30 ms, fixed

| metric | baseline |
|---|---|
| Caught (primary) | 44 / 100 |

Some prose between the tables.

### 35 ms, fixed

| metric | baseline |
|---|---|
| Caught (primary) | 41 / 100 |
"""


class ParsingTest(unittest.TestCase):
    def setUp(self):
        self.found = tables.tables(REPORT)

    def test_only_the_generated_tables_are_read(self):
        """What a person wrote above the marker is interpretation, not results."""
        self.assertEqual([heading for heading, _ in self.found],
                         ["Sensor noise", "30 ms, fixed", "35 ms, fixed"])

    def test_a_table_keeps_its_header_and_rows_but_not_the_rule(self):
        rows = dict(self.found)["Sensor noise"]
        self.assertEqual(rows[0], ["sigma_p", "A caught", "B caught"])
        self.assertEqual(rows[1], ["0 mm", "85 / 100", "85 / 100"])
        self.assertEqual(len(rows), 3)

    def test_a_table_belongs_to_the_nearest_heading_above_it(self):
        self.assertEqual(dict(self.found)["35 ms, fixed"][1], ["Caught (primary)", "41 / 100"])

    def test_several_tables_under_one_heading_are_numbered(self):
        """So neither overwrites the other's image."""
        doubled = f"{tables.MARKER}\n\n## One heading\n\n| a |\n|---|\n| 1 |\n\nprose\n\n| b |\n|---|\n| 2 |\n"
        self.assertEqual([heading for heading, _ in tables.tables(doubled)], ["One heading", "One heading 2"])

    def test_a_heading_becomes_a_filename(self):
        self.assertEqual(tables.slug("Close latency, with the trigger assuming 30 ms"),
                         "close-latency-with-the-trigger-assuming-30-ms")
        self.assertEqual(tables.slug("Tosses the nominal cell wasn't chosen from"),
                         "tosses-the-nominal-cell-wasn-t-chosen-from")


class DrawingTest(unittest.TestCase):
    def test_every_table_is_written_in_the_report_s_order(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "report.md"
            source.write_text(REPORT)
            paths = tables.make_all(source, Path(folder) / "tables")
            self.assertEqual([p.name for p in paths],
                             ["01-sensor-noise.png", "02-30-ms-fixed.png", "03-35-ms-fixed.png"])
            for path in paths:
                self.assertGreater(path.stat().st_size, 3000)

    def test_a_ragged_row_does_not_stop_the_drawing(self):
        """A row with fewer cells than the header is drawn blank, not crashed on."""
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "ragged.png"
            tables.draw("Ragged", [["a", "b", "c"], ["1"]], out)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
