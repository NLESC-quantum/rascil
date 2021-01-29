"""
Create a new branch, commit and push changes to it, then create a Merge Request.

This script was written to be executed in the compile_requirements GitLab CI job.
It uses python-gitlab and gitpython packages.
"""

import os
from configparser import NoSectionError

import gitlab
import logging

from git import Repo

log = logging.getLogger("rascil-logger")


class BranchManager:
    """
    Creat and check out new branches, commit and push changes.
    Find base branch.

    :param private_token: private access token of user with API and HTTP write access
    :param gitlab_user: user, whose token is used
    """

    def __init__(self, private_token, gitlab_user):
        self.private_token = private_token
        self.gitlab_user = gitlab_user
        self.repo = Repo(self._find_repository_root_dir())

    def _find_repository_root_dir(self):
        """
        Find the root directory of the repository.

        :return git_root: path of the root directory in string format
        """
        git_repo = Repo(".", search_parent_directories=True)
        git_root = git_repo.git.rev_parse("--show-toplevel")

        return git_root

    def set_git_config(self, username="Scheduled GitLab CI pipeline", email="<>"):
        """
        Set the user name and user email in git config.
        This is needed to commit changes.
        Use existing values if config file has a [user] section.
        """
        try:
            user_name = self.repo.config_reader(config_level="repository").get_value(
                "user", "name"
            )
            user_email = self.repo.config_reader(config_level="repository").get_value(
                "user", "email"
            )
        except NoSectionError:  # raised when the requested section doesn't exist in the config file
            self.repo.config_writer().set_value("user", "name", username).release()
            self.repo.config_writer().set_value("user", "email", email).release()
        else:
            log.info(
                f"User section already exists in gitconfig. "
                f"Defaulting to {user_name}, {user_email}."
            )

    def create_remote_with_token(self):
        """
        Create a new remote with the given user and access token.
        If remote already exists, delete it first, and recreate with correct token.

        :return remote_name: currently hard coded string "new_origin"
        """
        log.info("Creating new remote with gitlab user and its token.")

        remote_name = "new_origin"
        git_lab_push_url = (
            f"https://{self.gitlab_user}:{self.private_token}"
            f"@gitlab.com/ska-telescope/external/rascil.git"
        )

        if remote_name in self.repo.remotes:
            log.info(f"Remote {remote_name} already exists. Deleting.")
            self.repo.delete_remote(remote_name)

        self.repo.create_remote(remote_name, git_lab_push_url)

        return remote_name

    def commit_and_push_to_branch(self, new_branch_name, commit_message=None):
        # TODO: should we delete by default? how to make sure we don't delete something in use?
        self.set_git_config()

        if new_branch_name in self.repo.heads:
            self.repo.delete_head(new_branch_name, force=True)

        self.repo.git.checkout("-b", new_branch_name)
        print(f"Checked out new branch: {self.repo.active_branch}")

        self.repo.git.add(A=True)

        if not commit_message:
            self.repo.git.commit(m="Updated requirements")
        else:
            self.repo.git.commit(m=commit_message)

        remote_name = self.create_remote_with_token()
        self.repo.git.push("--set-upstream", remote_name, new_branch_name)

        return self.repo.active_branch.name

    def find_original_branch(self):
        try:
            original_branch = self.repo.active_branch.name

        except TypeError:
            detached_sha = self.repo.head.object.hexsha
            all_branches_with_sha = self.repo.git.branch(
                "-a", "--contains", detached_sha
            )
            original_branch = [
                x.strip() for x in all_branches_with_sha.split("\n") if "HEAD" not in x
            ][0]
            if "*" in original_branch:
                original_branch = original_branch.strip("* ")
            if "origin" in original_branch:
                original_branch = original_branch.split("origin/")[1]

        return original_branch

    def run_branch_manager(self, new_branch_name, message=None):
        if self.repo.index.diff(None) or self.repo.untracked_files:
            return self.commit_and_push_to_branch(
                new_branch_name, commit_message=message
            )

        else:
            log.info(
                "There weren't any changes detected. "
                "Not creating a new branch and new merge request."
            )
            return


class MergeRequest:
    def __init__(self, private_token):
        self.private_token = private_token
        self.gitlab_object = gitlab.Gitlab(
            "https://gitlab.com/", private_token, api_version="4"
        )

        # authenticate with GitLab
        self.gitlab_object.auth()

        self.rascil = self.gitlab_object.projects.get(19308749)

    def create_merge_request(self, source_branch, target_branch, mr_title):
        log.info(
            f"Creating merge request for branch {source_branch} to {target_branch}"
        )

        mr = self.rascil.mergerequests.create(
            {
                "source_branch": source_branch,
                "target_branch": target_branch,
                "title": mr_title,
            }
        )

        log.info("Merge request created.")
        return mr

    def assign_to_mr(self, mr, assignee):
        mr.assignee_id = assignee
        mr.save()
        log.info(f"Merge Request was assigned to user with id: {assignee}")


# def create_merge_request(gl, assignee, source_branch, target_branch):
#     rascil = gl.projects.get(19308749)
#
#     print(f"Creating merge request for branch {source_branch} to {target_branch}")
#     print(rascil.branches.list())
#     mr = rascil.mergerequests.create(
#         {
#             "source_branch": source_branch,
#             "target_branch": target_branch,
#             "title": "WIP: SIM-706: test MR",
#         }
#     )
#     mr.assignee_id = assignee
#     mr.save()
#     log.info("Merge request created and assigned.")


# def create_branch_and_commit(repo, private_token):
#     repo.config_writer().set_value(
#         "user", "name", "Scheduled GitLab CI pipeline"
#     ).release()
#     repo.config_writer().set_value("user", "email", "<>").release()
#
#     new_branch = "test-branch-to-update-reqs"
#
#     if new_branch in repo.heads:
#         repo.delete_head(new_branch, force=True)
#
#     remote_name = "new_origin"
#     git_lab_push_url = f"https://{os.environ['GITLAB_USER']}:{private_token}@gitlab.com/ska-telescope/external/rascil.git"
#     if remote_name in repo.remotes:
#         repo.delete_remote(remote_name)
#     repo.create_remote(remote_name, git_lab_push_url)
#
#     # current = repo.create_head(new_branch, origin.refs[new_branch]).set_tracking_branch(origin.refs[new_branch])
#     # current.checkout()
#     repo.git.checkout("-b", new_branch)
#     print(f"Checked out new branch: {repo.active_branch}")
#
#     repo.git.add(A=True)
#     repo.git.commit(m="test gitpython")
#
#     # print(origin)
#     # origin.fetch()
#     # origin.push(current, None, '--set-upstream')
#
#     repo.git.push("--set-upstream", remote_name, new_branch)
#     # log.info("Pushed new commits")
#
#     return repo.active_branch.name

os.environ["GITLAB_ACCESS_TOKEN"] = "test_token"
os.environ["GITLAB_USER"] = "test_user"
os.environ["GITLAB_ASSIGNEE_ID"] = "test_id"


def main():
    private_token = os.environ["GITLAB_ACCESS_TOKEN"]
    gitlab_user = os.environ["GITLAB_USER"]
    assignee_id = os.environ["GITLAB_ASSIGNEE_ID"]

    # repo = Repo(".")
    # print(f"Local git repository {repo}")

    # try:
    #     original_branch = repo.active_branch.name
    # except TypeError:
    #     detached_sha = repo.head.object.hexsha
    #     print(detached_sha)
    #     all_branches_with_sha = repo.git.branch("-a", "--contains", detached_sha)
    #     print(all_branches_with_sha)
    #     original_branch = [
    #         x.strip() for x in all_branches_with_sha.split("\n") if "HEAD" not in x
    #     ][0]
    #     if "*" in original_branch:
    #         original_branch = original_branch.strip("* ")
    #     if "origin" in original_branch:
    #         original_branch = original_branch.split("origin/")[1]

    branch_manager = BranchManager(private_token, gitlab_user)
    branch_manager.set_git_config()
    new_branch = branch_manager.run_branch_manager("test-branch-to-update-reqs")
    original_branch = branch_manager.find_original_branch()

    if new_branch:
        mr_object = MergeRequest(private_token)
        mr = mr_object.create_merge_request(
            new_branch, original_branch, "WIP: SIM-706: test MR"
        )
        mr_object.assign_to_mr(mr, assignee_id)

    else:
        log.info("No changes to commit.")


if __name__ == "__main__":
    main()
