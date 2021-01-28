import gitlab
from git import Repo
import logging

import os

log = logging.getLogger('rascil-logger')


def create_merge_request(gl, assignee, source_branch, target_branch):
    rascil = gl.projects.get(19308749)

    print(f"Creating merge request for branch {source_branch} to {target_branch}")
    print(rascil.branches.list())
    mr = rascil.mergerequests.create({'source_branch': source_branch,
                                      'target_branch': target_branch,
                                      'title': 'WIP: SIM-706: test MR'})
    mr.assignee_id = assignee
    log.info("Merge request created and assigned.")


def create_branch_and_commit(repo, private_token):
    repo.config_writer().set_value("user", "name", "Scheduled GitLab CI pipeline").release()
    repo.config_writer().set_value("user", "email", "<>").release()

    new_branch = 'test-branch-to-update-reqs'

    if new_branch in repo.heads:
        repo.delete_head(new_branch, force=True)

    remote_name = "new_origin"
    git_lab_push_url = f"https://{os.environ['GITLAB_USER']}:{private_token}@gitlab.com/ska-telescope/external/rascil.git"
    if remote_name in repo.remotes:
        repo.delete_remote(remote_name)
    repo.create_remote(remote_name, git_lab_push_url)

    # current = repo.create_head(new_branch, origin.refs[new_branch]).set_tracking_branch(origin.refs[new_branch])
    # current.checkout()
    repo.git.checkout("-b", new_branch)
    print(f"Checked out new branch: {repo.active_branch}")

    repo.git.add(A=True)
    repo.git.commit(m='test gitpython')

    # print(origin)
    # origin.fetch()
    # origin.push(current, None, '--set-upstream')

    repo.git.push('--set-upstream', remote_name, new_branch)
    # log.info("Pushed new commits")

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
        print(original_branch)
        repo.git.checkout(original_branch)
        repo.git.pull()

    print(original_branch)
    if repo.index.diff(None) or repo.untracked_files:
        # new_branch = create_branch_and_commit(repo, private_token)
        new_branch = 'test-branch-to-update-reqs'
        create_merge_request(gl, assignee_id, new_branch, original_branch)

    else:
        log.info("No changes to commit.")


if __name__ == '__main__':
    main()