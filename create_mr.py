import gitlab
from git import Repo
import logging

import os

log = logging.getLogger('rascil-logger')


def create_merge_request(gl, assignee, source_branch, target_branch):
    log.info(f"Creating merge request for branch {source_branch} to {target_branch}")
    mr = gl.mergerequests.create({'source_branch': source_branch,
                                  'target_branch': target_branch,
                                  'title': 'WIP: SIM-706: test MR'})
    mr.assignee_id = assignee
    log.info("Merge request created and assigned.")


def create_branch_and_commit(repo):
    repo.config_writer().set_value("user", "name", "Scheduled GitLab CI pipeline").release()
    repo.config_writer().set_value("user", "email", "<>").release()

    new_branch = 'test-branch-to-update-reqs'
    current = repo.create_head(new_branch)
    current.checkout()
    log.info(f"Checked out new branch: {repo.active_branch}")

    repo.git.add(A=True)
    repo.git.commit(m='test gitpython')
    repo.git.push('--set-upstream', 'origin', current)
    log.info("Pushed new commits")

    return repo.active_branch.name


def main():
    origin = "https://gitlab.com/"
    private_token = os.environ["GITLAB_ACCESS_TOKEN"]

    gl = gitlab.Gitlab(origin, private_token, api_version='4')
    gl.auth()

    repo = Repo(".")
    log.info(f"Local git repository {repo}")

    assignee_id = os.environ["GITLAB_ASSIGNEE_ID"]

    try:
        original_branch = repo.active_branch.name
    except TypeError:
        detached_sha = repo.head.object.hexsha
        original_branch = repo.git.branch('--contains', detached_sha).split('*')[1].strip()
        # repo.git.checkout(original_branch)

    if repo.index.diff(None) or repo.untracked_files:
        new_branch = create_branch_and_commit(repo)
        create_merge_request(gl, assignee_id, new_branch, original_branch)

    else:
        log.info("No changes to commit.")


if __name__ == '__main__':
    main()