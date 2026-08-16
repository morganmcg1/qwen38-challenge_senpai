#!/usr/bin/env python3
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("submit-official.sh").resolve()
BENCHMARK_ID = "5d1ee4d7-80bd-4555-b182-6505f26ef495"
BENCHMARK_NAME = "eigenlabs/qwen38-challenge"


def run(argv, cwd, *, env=None, check=True):
    result = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if check and result.returncode:
        raise AssertionError(f"{argv!r} failed ({result.returncode}):\n{result.stdout}")
    return result


class OfficialSubmissionGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="qwen38-submit-guard-")
        self.root = Path(self.temporary.name)
        self.origin = self.root / "origin.git"
        self.upstream = self.root / "upstream.git"
        self.repository = self.root / "candidate"
        self.bin = self.root / "bin"
        self.log = self.root / "yukon-argv"
        self.script = self.root / "submit-official.sh"

        run(["git", "init", "--bare", "-q", self.origin], self.root)
        run(["git", "init", "--bare", "-q", self.upstream], self.root)
        run(["git", "init", "-q", self.repository], self.root)
        run(["git", "config", "user.name", "Test"], self.repository)
        run(["git", "config", "user.email", "test@example.com"], self.repository)
        run(["git", "branch", "-M", "main"], self.repository)

        (self.repository / "Sources").mkdir()
        (self.repository / "benchmark.json").write_text(
            '{"schemaVersion":1,"editablePaths":["Sources"]}\n',
            encoding="utf-8",
        )
        (self.repository / "Sources/kernel.swift").write_text("base\n", encoding="utf-8")
        (self.repository / "research.md").write_text("notes\n", encoding="utf-8")
        (self.repository / "AGENTS.md").write_text(
            "# Test Agent Guide\n\nOrganizer text.\n", encoding="utf-8"
        )
        (self.repository / ".gitignore").write_text("base-ignore\n", encoding="utf-8")
        run(["git", "add", "."], self.repository)
        run(["git", "commit", "-qm", "organizer base"], self.repository)
        self.organizer_base = run(
            ["git", "rev-parse", "HEAD"], self.repository
        ).stdout.strip()
        run(["git", "remote", "add", "origin", self.origin], self.repository)
        run(["git", "remote", "add", "upstream", self.upstream], self.repository)
        run(["git", "config", "yukon.benchmark-id", BENCHMARK_ID], self.repository)
        run(["git", "config", "yukon.benchmark-name", BENCHMARK_NAME], self.repository)
        run(["git", "config", "yukon.source-url", str(self.upstream)], self.repository)
        run(["git", "config", "yukon.source-branch", "main"], self.repository)
        run(["git", "push", "-q", "upstream", "main"], self.repository)
        run(["git", "remote", "set-url", "--push", "upstream", "DISABLED"], self.repository)

        (self.repository / "senpai").mkdir()
        (self.repository / "AGENTS.md").write_text(
            "<!-- SENPAI-CAMPAIGN-BEGIN -->\n"
            "# Campaign Agent Guide\n\nCampaign-owned text.\n"
            "<!-- SENPAI-CAMPAIGN-END -->\n",
            encoding="utf-8",
        )
        (self.repository / ".gitignore").write_text(
            "base-ignore\n"
            "# SENPAI-CAMPAIGN-BEGIN\n"
            "senpai/.env*\n"
            "# SENPAI-CAMPAIGN-END\n",
            encoding="utf-8",
        )
        (self.repository / "senpai/frontier-state.json").write_text(
            "{\n"
            '  "schemaVersion": 1,\n'
            '  "observedAt": "2026-08-16T00:00:00Z",\n'
            f'  "benchmark": {{"id": "{BENCHMARK_ID}", "name": "{BENCHMARK_NAME}"}},\n'
            f'  "organizer": {{"url": "{self.upstream}", "branch": "main", '
            f'"syncedCommit": "{self.organizer_base}"}},\n'
            f'  "promotedSubmission": {{"id": "submission-id", '
            f'"sourceRef": "{self.organizer_base}", "score": 1.25}}\n'
            "}\n",
            encoding="utf-8",
        )
        run(
            ["git", "add", "senpai/frontier-state.json", "AGENTS.md", ".gitignore"],
            self.repository,
        )
        run(["git", "commit", "-qm", "campaign base"], self.repository)
        self.base = run(["git", "rev-parse", "HEAD"], self.repository).stdout.strip()
        run(["git", "push", "-qu", "origin", "main"], self.repository)

        run(["git", "switch", "-qc", "experiment"], self.repository)
        (self.repository / "Sources/kernel.swift").write_text(
            "candidate\n", encoding="utf-8"
        )
        run(["git", "add", "Sources/kernel.swift"], self.repository)
        run(["git", "commit", "-qm", "candidate"], self.repository)
        self.candidate = run(
            ["git", "rev-parse", "HEAD"], self.repository
        ).stdout.strip()

        self.bin.mkdir()
        fake_cli = self.bin / "yukon"
        fake_cli.write_text(
            '#!/bin/sh\nprintf "%s\\n" "$@" > "$YUKON_FAKE_LOG"\n',
            encoding="utf-8",
        )
        fake_cli.chmod(0o755)

        script_text = SCRIPT.read_text(encoding="utf-8")
        script_text = script_text.replace(
            'EXPECTED_ORIGIN_URL="https://github.com/morganmcg1/qwen38-challenge_senpai.git"',
            f'EXPECTED_ORIGIN_URL="{self.origin}"',
        ).replace(
            'EXPECTED_UPSTREAM_URL="https://github.com/Layr-Labs/qwen-3.8-mtp-challenge"',
            f'EXPECTED_UPSTREAM_URL="{self.upstream}"',
        )
        self.script.write_text(script_text, encoding="utf-8")
        self.script.chmod(0o755)
        self.environment = os.environ | {
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "YUKON_FAKE_LOG": str(self.log),
        }

    def tearDown(self):
        self.temporary.cleanup()

    def submit(self, *arguments):
        default = ("--model", "senpai", "--note-file", "submission-note.md")
        return self.submit_with_base(self.base, *(arguments or default))

    def submit_with_base(self, base, *arguments):
        return run(
            ["/bin/bash", self.script, base, *arguments],
            self.repository,
            env=self.environment,
            check=False,
        )

    def publish(self, remote, mutate, message):
        checkout = self.root / f"publisher-{remote}"
        remote_path = self.origin if remote == "origin" else self.upstream
        run(["git", "clone", "-q", "--branch", "main", remote_path, checkout], self.root)
        run(["git", "config", "user.name", "Test"], checkout)
        run(["git", "config", "user.email", "test@example.com"], checkout)
        mutate(checkout)
        run(["git", "add", "."], checkout)
        run(["git", "commit", "-qm", message], checkout)
        run(["git", "push", "-q", "origin", "main"], checkout)

    def test_current_base_submits_with_exact_passthrough_arguments(self):
        result = self.submit()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(
            self.log.read_text(encoding="utf-8").splitlines(),
            ["submit", "--model", "senpai", "--note-file", "submission-note.md"],
        )

    def test_dirty_research_is_allowed_but_dirty_submission_surface_is_not(self):
        (self.repository / "research.md").write_text("working notes\n", encoding="utf-8")
        allowed = self.submit()
        self.assertEqual(allowed.returncode, 0, allowed.stdout)

        self.log.unlink()
        (self.repository / "Sources/kernel.swift").write_text(
            "uncommitted\n", encoding="utf-8"
        )
        refused = self.submit()
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("commit or discard changes", refused.stdout)
        self.assertFalse(self.log.exists())

    def test_campaign_main_source_advance_invalidates_old_base(self):
        self.publish(
            "origin",
            lambda checkout: (checkout / "Sources/kernel.swift").write_text(
                "new frontier\n", encoding="utf-8"
            ),
            "advance frontier",
        )
        refused = self.submit()
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("submitted snapshot differs", refused.stdout)
        self.assertFalse(self.log.exists())

    def test_docs_only_campaign_advance_does_not_invalidate_base(self):
        self.publish(
            "origin",
            lambda checkout: (checkout / "senpai/frontier.txt").write_text(
                "research note\n", encoding="utf-8"
            ),
            "document frontier",
        )
        allowed = self.submit()
        self.assertEqual(allowed.returncode, 0, allowed.stdout)
        self.assertTrue(self.log.exists())

    def test_campaign_owned_agents_advance_does_not_invalidate_base(self):
        self.publish(
            "origin",
            lambda checkout: (checkout / "AGENTS.md").write_text(
                "<!-- SENPAI-CAMPAIGN-BEGIN -->\n"
                "# Updated Campaign Guide\n\nQwen-only instructions.\n"
                "<!-- SENPAI-CAMPAIGN-END -->\n",
                encoding="utf-8",
            ),
            "update campaign agent guide",
        )
        allowed = self.submit()
        self.assertEqual(allowed.returncode, 0, allowed.stdout)
        self.assertTrue(self.log.exists())

    def test_agents_content_outside_campaign_block_is_rejected(self):
        self.publish(
            "origin",
            lambda checkout: (checkout / "AGENTS.md").write_text(
                "<!-- SENPAI-CAMPAIGN-BEGIN -->\n"
                "# Campaign Guide\n"
                "<!-- SENPAI-CAMPAIGN-END -->\n"
                "Organizer text reintroduced.\n",
                encoding="utf-8",
            ),
            "reintroduce organizer agent prose",
        )
        refused = self.submit()
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("must be wholly campaign-owned", refused.stdout)
        self.assertFalse(self.log.exists())

    def test_campaign_agents_symlink_is_rejected(self):
        def replace_with_symlink(checkout):
            path = checkout / "AGENTS.md"
            path.unlink()
            path.symlink_to("benchmark.json")

        self.publish(
            "origin",
            replace_with_symlink,
            "replace campaign guide with symlink",
        )
        refused = self.submit()
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("must be a regular file", refused.stdout)
        self.assertFalse(self.log.exists())

    def test_campaign_trusted_surface_drift_is_rejected(self):
        self.publish(
            "origin",
            lambda checkout: (checkout / "workflow.yml").write_text(
                "unreviewed trusted change\n", encoding="utf-8"
            ),
            "drift campaign harness",
        )
        refused = self.submit()
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("campaign main has unreviewed trusted drift", refused.stdout)
        self.assertFalse(self.log.exists())

    def test_organizer_contract_advance_requires_sync(self):
        self.publish(
            "upstream",
            lambda checkout: (checkout / "benchmark.json").write_text(
                '{"schemaVersion":1,"editablePaths":["Sources"],"newRule":true}\n',
                encoding="utf-8",
            ),
            "change contract",
        )
        refused = self.submit()
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("organizer benchmark.json differs", refused.stdout)
        self.assertFalse(self.log.exists())

    def test_organizer_trusted_surface_advance_requires_sync(self):
        self.publish(
            "upstream",
            lambda checkout: (checkout / "workflow.yml").write_text(
                "trusted change\n", encoding="utf-8"
            ),
            "change trusted harness",
        )
        refused = self.submit()
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("trusted surface advanced", refused.stdout)
        self.assertFalse(self.log.exists())

    def test_organizer_editable_only_advance_does_not_block_submission(self):
        self.publish(
            "upstream",
            lambda checkout: (checkout / "Sources/kernel.swift").write_text(
                "another public candidate\n", encoding="utf-8"
            ),
            "validate another candidate",
        )
        allowed = self.submit()
        self.assertEqual(allowed.returncode, 0, allowed.stdout)
        self.assertTrue(self.log.exists())

    def test_model_note_and_track_arguments_fail_closed(self):
        missing_model = self.submit_with_base(
            self.base, "--note-file", "submission-note.md"
        )
        self.assertEqual(missing_model.returncode, 2)
        self.assertIn("--model", missing_model.stdout)

        for model in ("GPT 5.6 Sol", "Senpai", " senpai"):
            with self.subTest(model=model):
                wrong_model = self.submit_with_base(
                    self.base,
                    "--model",
                    model,
                    "--note-file",
                    "submission-note.md",
                )
                self.assertEqual(wrong_model.returncode, 2)
                self.assertIn('must be --model "senpai"', wrong_model.stdout)

        duplicate_model = self.submit_with_base(
            self.base,
            "--model",
            "senpai",
            "--model=senpai",
            "--note-file",
            "submission-note.md",
        )
        self.assertEqual(duplicate_model.returncode, 2)
        self.assertIn("exactly once", duplicate_model.stdout)

        missing_note = self.submit_with_base(self.base, "--model", "senpai")
        self.assertEqual(missing_note.returncode, 2)
        self.assertIn("--note", missing_note.stdout)

        track = self.submit_with_base(
            self.base,
            "--track",
            "qwen3.8-27b-mtp-v1",
            "--model",
            "senpai",
            "--note-file",
            "submission-note.md",
        )
        self.assertEqual(track.returncode, 2)
        self.assertIn("schema v1", track.stdout)

        benchmark = self.submit_with_base(
            self.base,
            "someone/another-benchmark",
            "--model",
            "senpai",
            "--note-file",
            "submission-note.md",
        )
        self.assertEqual(benchmark.returncode, 2)
        self.assertIn("explicit benchmark", benchmark.stdout)

        unknown = self.submit_with_base(
            self.base,
            "--unknown",
            "value",
            "--model",
            "senpai",
            "--note-file",
            "submission-note.md",
        )
        self.assertEqual(unknown.returncode, 2)
        self.assertIn("unsupported Yukon option", unknown.stdout)

    def test_committed_contract_change_is_rejected(self):
        (self.repository / "benchmark.json").write_text(
            '{"schemaVersion":1,"editablePaths":["Sources","research.md"]}\n',
            encoding="utf-8",
        )
        run(["git", "add", "benchmark.json"], self.repository)
        run(["git", "commit", "-qm", "poison submission contract"], self.repository)

        refused = self.submit()
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("benchmark.json differs", refused.stdout)
        self.assertFalse(self.log.exists())

    def test_skip_worktree_entry_is_rejected(self):
        run(
            ["git", "update-index", "--skip-worktree", "Sources/kernel.swift"],
            self.repository,
        )
        (self.repository / "Sources/kernel.swift").write_text(
            "hidden change\n", encoding="utf-8"
        )

        refused = self.submit()
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("skip-worktree/assume-unchanged", refused.stdout)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
