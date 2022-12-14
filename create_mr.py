"""
Create a new branch, commit and push changes to it, then create a Merge Request.

This script was written to be executed in the compile_requirements GitLab CI job.
It uses python-gitlab and gitpython packages.
"""

import os
import sys
import gitlab
import logging
from configparser import NoSectionError
from git import Repo

log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


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
        """
        Create new branch, unless it already exists: if it does, delete first.
        Add, commit, and push changes to this new branch.

        :param new_branch_name: name of the new branch to be checked out
        :param commit_message: message to add to the commit; if None, use default message
        """
        self.set_git_config()

        # TODO: should we delete by default? how to make sure we don't delete something in use?
        if new_branch_name in self.repo.heads:
            self.repo.delete_head(new_branch_name, force=True)

        self.repo.git.checkout("-b", new_branch_name)
        log.info(f"Checked out new branch: {self.repo.active_branch}")

        self.repo.git.add(A=True)

        if not commit_message:
            self.repo.git.commit(m="ORC-805: Updated requirements")
        else:
            self.repo.git.commit(m=commit_message)

        remote_name = self.create_remote_with_token()
        self.repo.git.push("--set-upstream", remote_name, new_branch_name)

        return self.repo.active_branch.name

    def run_branch_manager(self, new_branch_name, message=None):
        """
        Execute the process of creating a new branch,
        committing changes, and pushing to new branch,
        if branch is dirty, i.e. there are files that changed.
        Otherwise log info and return without making changes.

        :param new_branch_name: new branch name to checkout if changes are present
        :param message: commit message to use if changes are committed
        """
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
    """
    Create and manage a GitLab Merge Request.

    :param private_token: private access token of user with API and HTTP write access
    """

    def __init__(self, private_token):
        self.private_token = private_token
        self.gitlab_object = gitlab.Gitlab(
            "https://gitlab.com/", private_token, api_version="4"
        )

        # authenticate with GitLab
        self.gitlab_object.auth()

        self.rascil = self.gitlab_object.projects.get(19308749)

    def create_merge_request(self, source_branch, target_branch, mr_title):
        """
        Create a new merge request

        :param source_branch: branch to be merged
        :param target_branch: branch to be merged into
        :param mr_title: title of the merge request
        """
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

    def udpate_mr_settings(self, mr, assignees, reviewers):
        """
        Update merge request settings:
            - assign MR
            - add reviewers

        :param mr: merge request object
        :param assignees: unique GitLab User ID of the person to assign to
                          to find yours, navigate to Settings --> Main Settings --> User ID
        :param reviewers: unique GitLab User ID of the person to set as reviewer
        """
        mr.assignee_ids = assignees
        mr.reviewer_ids = reviewers
        mr.save()
        log.info(f"Merge Request was assigned to user(s) with id(s): {assignees}")
        log.info(f"Review request was sent to user(s) with id(s): {reviewers}")


def main():
    private_token = os.environ["PROJECT_ACCESS_TOKEN"]
    gitlab_user = os.environ["PROJECT_TOKEN_USER"]
    assignee_ids = os.environ["GITLAB_ASSIGNEE_ID"]
    reviewer_ids = os.environ["GITLAB_REVIEWER_ID"]
    new_branch_name = "orc-805-scheduled-update-requirements"

    branch_manager = BranchManager(private_token, gitlab_user)
    branch_manager.set_git_config()
    new_branch = branch_manager.run_branch_manager(new_branch_name)

    if new_branch:
        original_branch = os.environ["CI_COMMIT_BRANCH"]
        mr_title = "Draft: ORC-805: Update requirements - to be actioned before next scheduled run"
        mr_object = MergeRequest(private_token)
        mr = mr_object.create_merge_request(new_branch, original_branch, mr_title)
        mr_object.udpate_mr_settings(
            mr, assignee_ids.split(","), reviewer_ids.split(",")
        )

    else:
        log.info("No changes to commit.")


if __name__ == "__main__":
    main()
