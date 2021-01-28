import gitlab
from git import Repo

import os


def create_merge_request(gl, assignee):
    mr = gl.mergerequests.create({'source_branch': 'cool_feature',
                               'target_branch': 'master',
                               'title': 'merge cool feature',
                               'labels': ['label1', 'label2']})
    mr.assignee_id = assignee


def main():
    # origin = "https://gitlab.com/"
    private_token = os.environ["GITLAB_TOKEN"]
    #
    # gl = gitlab.Gitlab(origin, private_token, api_version='4')
    # gl.auth()

    repo = Repo(".")

    assignee_id = "7801632"  # GH

    new_branch = 'test-branch-to-update-reqs'
    current = repo.create_head(new_branch)
    current.checkout()
    # master = repo.heads.master
    # repo.git.pull('origin', master)

    if repo.index.diff(None) or repo.untracked_files:

        repo.git.add(A=True)
        repo.git.commit(m='test gitpython')
        repo.git.push('--set-upstream', 'origin', current)
        print('git push')
    else:
        print('no changes')

if __name__ == '__main__':
    main()