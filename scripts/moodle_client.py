"""
Wrapper around Moodle's Web Services REST API.

Requires a webservice token with these Moodle functions enabled in the
external service (Site Administration > Server > Web services):
  - core_course_get_contents
  - mod_forum_get_forum_discussions_paginated (optional, for forum content)
  - core_course_get_courses (optional, for course metadata)

See: https://docs.moodle.org/dev/Web_service_API_functions
"""
import re
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup


class MoodleClient:
    def __init__(self, base_url: str, token: str):
        self.endpoint = f"{base_url.rstrip('/')}/webservice/rest/server.php"
        self.token = token

    def _call(self, wsfunction: str, **params) -> Any:
        query = {
            "wstoken": self.token,
            "wsfunction": wsfunction,
            "moodlewsrestformat": "json",
            **params,
        }
        resp = requests.get(self.endpoint, params=query, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("exception"):
            raise RuntimeError(
                f"Moodle API error calling {wsfunction}: {data.get('message')}"
            )
        return data

    def get_course_contents(self, course_id: int) -> List[Dict]:
        """Returns the raw section/module structure for a course."""
        return self._call("core_course_get_contents", courseid=course_id)

    def get_forum_discussions(self, forum_id: int) -> List[Dict]:
        result = self._call(
            "mod_forum_get_forum_discussions_paginated", forumid=forum_id
        )
        return result.get("discussions", [])

    def extract_text_content(self, course_id: int) -> List[Dict[str, str]]:
        """
        Walks course_contents and produces a flat list of
        {module_name, source_type, source_url, text} ready for chunking.

        Handles: pages, labels, and other modules with inline HTML
        `description`/`intro` content directly. File resources (PDF,
        DOCX, PPTX) are listed but not parsed here — see the TODO below.
        """
        sections = self.get_course_contents(course_id)
        items = []

        for section in sections:
            for module in section.get("modules", []):
                mod_name = module.get("name", "Untitled")
                mod_type = module.get("modname", "unknown")
                mod_url = module.get("url", "")

                # Inline text content (pages, labels, descriptions)
                description = module.get("description", "")
                if description:
                    text = self._html_to_text(description)
                    if text.strip():
                        items.append(
                            {
                                "module_name": mod_name,
                                "source_type": mod_type,
                                "source_url": mod_url,
                                "text": text,
                            }
                        )

                # File contents (page content, etc. stored as "contents")
                for content in module.get("contents", []):
                    if content.get("type") == "content" and content.get("content"):
                        # Some modules (e.g. mod_page) embed HTML content directly
                        text = self._html_to_text(content["content"])
                        if text.strip():
                            items.append(
                                {
                                    "module_name": mod_name,
                                    "source_type": mod_type,
                                    "source_url": content.get("fileurl", mod_url),
                                    "text": text,
                                }
                            )
                    elif content.get("type") == "file":
                        # TODO: download content["fileurl"] + "&token=" + self.token
                        # and extract text with pypdf / python-docx / python-pptx
                        # depending on content["mimetype"]. Skipped in this
                        # scaffold — see README "Known gaps".
                        pass

        return items

    @staticmethod
    def _html_to_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text
