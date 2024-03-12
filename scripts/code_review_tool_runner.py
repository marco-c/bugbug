# -*- coding: utf-8 -*-
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

import os

from bugbug import phabricator
from bugbug.tools import code_review
from bugbug.utils import get_secret


def main():
    os.environ["OPENAI_API_KEY"] = get_secret("OPENAI_API_KEY")

    phabricator.set_api_key(
        get_secret("PHABRICATOR_URL"), get_secret("PHABRICATOR_TOKEN")
    )

    code_review_tool = code_review.PhabricatorCodeReviewTool()

    revisions = phabricator.get(rev_ids=[203997])
    for revision in revisions:
        patch = phabricator.PHABRICATOR_API.load_raw_diff(revision["fields"]["diffID"])
        print(patch)
        code_review_tool.run(patch)
        input()


if __name__ == "__main__":
    main()
