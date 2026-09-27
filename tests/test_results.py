import json
import subprocess

from gptlab.runner.results import ResultsRepo, authed_url


def bare_repo(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    return str(remote)


def make(tmp_path, remote, name):
    return ResultsRepo(remote, None, tmp_path / name, "results", "Tester", "t@example.com", log=lambda m: None)


def write_status(state):
    def fill(folder):
        (folder / "status.json").write_text(json.dumps({"state": state}))
    return fill


def test_create_branch_publish_and_read(tmp_path):
    remote = bare_repo(tmp_path)
    a = make(tmp_path, remote, "a")
    a.sync()
    assert a.statuses() == {}
    a.publish("job1", write_status("running"), "claim job1")
    b = make(tmp_path, remote, "b")
    b.sync()
    assert b.statuses() == {"job1": {"state": "running"}}
    log = subprocess.run(["git", "log", "--format=%an %s", "results"], cwd=remote, capture_output=True, text=True)
    assert "Tester claim job1" in log.stdout


def test_two_sessions_push_without_losing_each_other(tmp_path):
    remote = bare_repo(tmp_path)
    a, b = make(tmp_path, remote, "a"), make(tmp_path, remote, "b")
    a.sync()
    b.sync()  # both clones are now at the same commit
    a.publish("job_a", write_status("done"), "a")
    b.publish("job_b", write_status("done"), "b")  # b's clone is behind; must not drop job_a
    c = make(tmp_path, remote, "c")
    c.sync()
    assert c.statuses() == {"job_a": {"state": "done"}, "job_b": {"state": "done"}}


def test_claim_check_stops_a_second_claim(tmp_path):
    remote = bare_repo(tmp_path)
    a, b = make(tmp_path, remote, "a"), make(tmp_path, remote, "b")
    a.sync()
    b.sync()

    def free(status):
        return status is None or status["state"] != "running"

    assert a.publish("job", write_status("running"), "a claims", check=free)
    assert not b.publish("job", write_status("running"), "b claims", check=free)


def test_token_is_put_in_url_and_hidden_in_errors(tmp_path):
    assert authed_url("https://github.com/o/r.git", "SECRET") == "https://x-access-token:SECRET@github.com/o/r.git"
    r = ResultsRepo("https://github.com/o/r.git", "SECRET", tmp_path / "x", "results", "n", "e")
    assert r._redact("url https://x-access-token:SECRET@github.com") == "url https://x-access-token:***@github.com"
