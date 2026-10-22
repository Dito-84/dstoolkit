"""Turn pytest JUnit XML failures into GitHub Actions error annotations.

Annotations show on the pull request and are readable through the GitHub API,
which makes browser-test failures easy to diagnose without opening the logs.
"""

import sys
import xml.etree.ElementTree as ET


def escape(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "").replace("\n", "%0A")


def main(path: str) -> None:
    root = ET.parse(path).getroot()
    for case in root.iter("testcase"):
        for problem in list(case.findall("failure")) + list(case.findall("error")):
            detail = (problem.get("message") or "") + "\n" + (problem.text or "")
            print(f"::error title={case.get('name')}::{escape(detail[-6000:])}")


if __name__ == "__main__":
    main(sys.argv[1])
