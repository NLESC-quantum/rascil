import pytest

from configparser import NoSectionError
from unittest.mock import patch, Mock, call, PropertyMock
from git import GitConfigParser, Repo

from create_mr import BranchManager


@pytest.fixture
def manager_fixture():
    private_token = "abc"
    gitlab_user = "me"
    manager = BranchManager(private_token, gitlab_user)

    return manager


@patch.object(GitConfigParser, "get_value", Mock(side_effect=NoSectionError("Error")))
@patch.object(GitConfigParser, "set_value")
# if not mocked, it'll think there is a lock on the config file
@patch.object(GitConfigParser, "_acquire_lock", Mock())
def test_set_git_config_default_values(fake_set, manager_fixture):
    """
    BranchManager.set_git_config() uses the default user and email values
    to set the git config, if it doesn't find existing user settings
    (i.e. NoSectionError is raised), and input values are not provided.
    """
    expected_calls = [
        call("user", "name", "Scheduled GitLab CI pipeline"),
        call("user", "email", "<>"),
    ]

    manager_fixture.set_git_config()
    assert fake_set.call_count == 2
    assert fake_set.call_args_list == expected_calls


@patch.object(GitConfigParser, "get_value", Mock(side_effect=NoSectionError("Error")))
@patch.object(GitConfigParser, "set_value")
# if not mocked, it'll think there is a lock on the config file
@patch.object(GitConfigParser, "_acquire_lock", Mock())
def test_set_git_config_custom_values(fake_set, manager_fixture):
    """
    BranchManager.set_git_config() uses the the provided user and email values
    to set the git config, if it doesn't find existing user settings
    (i.e. NoSectionError is raised).
    """
    user = "new-user"
    email = "my-new-email"

    expected_calls = [call("user", "name", user), call("user", "email", email)]

    manager_fixture.set_git_config(user, email)
    assert fake_set.call_count == 2
    assert fake_set.call_args_list == expected_calls


@patch.object(GitConfigParser, "get_value", Mock(side_effect=["my-name", "my-email"]))
def test_set_git_config_user_exists(manager_fixture):
    """
    BranchManager.set_git_config() will find that the [user] configuration
    is already set (Mocked by "fake_value") and hence will use those settings,
    and not update them.
    """
    with patch("logging.Logger.info") as mock_log:
        manager_fixture.set_git_config()
        mock_log.assert_called_with(
            "User section already exists in gitconfig. "
            "Defaulting to my-name, my-email."
        )


@pytest.mark.parametrize(
    "remotes_list, expected_delete_call", [([], False), (["new_origin"], True)]
)
@patch.object(Repo, "create_remote")
@patch.object(Repo, "delete_remote")
def test_create_remote_with_token_remote_does_not_exist(
    mock_delete_remote,
    mock_create_remote,
    remotes_list,
    expected_delete_call,
    manager_fixture,
):
    """
    Test #1:
    BranchManager.create_remote_with_token() does not call Repo.delete_remote,
    because the remote we want to create does not already exist.

    Test #2:
    BranchManager.create_remote_with_token() calls Repo.delete_remote,
    because the remote ("new_origin") we want to create already exists.

    Then, it creates the new remote with the give user and token.
    """
    with patch.object(Repo, "remotes", remotes_list):
        manager_fixture.create_remote_with_token()

        if not expected_delete_call:
            mock_delete_remote.assert_not_called()
        else:
            mock_delete_remote.assert_called_once()

        mock_create_remote.assert_called_with(
            "new_origin", "https://me:abc@gitlab.com/ska-telescope/external/rascil.git"
        )


@patch.object(BranchManager, "set_git_config", Mock())
@patch.object(BranchManager, "create_remote_with_token", Mock())
@patch.object(Repo, "delete_head")
@patch("git.Repo.git", new_callable=PropertyMock)
class TestCommitAndPushToBranch:
    @patch.object(Repo, "heads", [])
    def test_default_behaviour(self, mock_git, mock_delete_head, manager_fixture):
        """
        Given
            BranchManager.commit_and_push_to_branch
        When
            - the new branch name is not in repo.heads
            - custom commit message is not provided
        Then
            - Repo.delete_head is not called
            - new branch is checked out
            - the commit message is the default message
            - git.push is called
        """
        new_branch = "my-new-branch"

        manager_fixture.commit_and_push_to_branch(new_branch)

        mock_delete_head.assert_not_called()
        mock_git.return_value.checkout.assert_called_with("-b", new_branch)
        mock_git.return_value.commit.assert_called_with(m="Updated requirements")
        mock_git.return_value.push.assert_called()

    @patch.object(Repo, "heads", ["my-new-branch"])
    def test_branch_in_heads(self, mock_git, mock_delete_head, manager_fixture):
        """
        Given
            BranchManager.commit_and_push_to_branch
        When
            - the new branch name is already in repo.heads
            - custom commit message is not provided
        Then
            - Repo.delete_head is called to remove the branch from heads
            - new branch is checked out
            - the commit message is the default message
            - git.push is called
        """
        new_branch = "my-new-branch"

        manager_fixture.commit_and_push_to_branch(new_branch)

        mock_delete_head.assert_called_once()
        mock_git.return_value.checkout.assert_called_with("-b", new_branch)
        mock_git.return_value.commit.assert_called_with(m="Updated requirements")
        mock_git.return_value.push.assert_called()

    @patch.object(Repo, "heads", [])
    def test_custom_commit_message(self, mock_git, mock_delete_head, manager_fixture):
        """
        Given
            BranchManager.commit_and_push_to_branch
        When
            - a custom commit message is provided
        Then
            - the commit message is updated to be the custom message
        """
        new_branch = "my-new-branch"
        commit_message = "my custom message"

        manager_fixture.commit_and_push_to_branch(
            new_branch, commit_message=commit_message
        )
        mock_git.return_value.commit.assert_called_with(m=commit_message)


@patch.object(Repo, "active_branch")
def test_find_original_branch_active(mock_active_branch, manager_fixture):
    """
    We are on an active branch, which is returned
    """
    mock_active_branch.return_value = Mock()
    type(mock_active_branch).name = PropertyMock(return_value="working-branch")

    result = manager_fixture.find_original_branch()
    assert result == "working-branch"


@pytest.mark.parametrize(
    "all_branches_string, expected_branch",
    [
        ("some-branch-name", "some-branch-name"),
        ("branch-to-return\nHEAD do not return", "branch-to-return"),
        ("branch with * \nHEAD", "branch with"),
        ("HEAD leave\n more HEAD\n test/origin/in-branch-name", "in-branch-name"),
    ],
)
@patch.object(Repo, "active_branch")
@patch.object(Repo, "head", Mock())
@patch("git.Repo.git", new_callable=PropertyMock)
def test_find_original_branch_detached(
    mock_git, mock_active_branch, all_branches_string, expected_branch, manager_fixture
):
    """
    We are on an detached head; code finds the source branch.
    Test cases:
        1. only one branch found and that is returned as is
        2. two branches found, one contains the "HEAD" word; only the one without "HEAD" is returned
        3. two branches found, the one without "HEAD" is returned, stripped of "* "
        4. two branches found, the one without "HEAD" is returned, stripped of "test/origin/"
    """
    mock_active_branch.return_value = Mock()
    type(mock_active_branch).name = PropertyMock(side_effect=TypeError("Error"))
    mock_git.return_value = Mock(branch=Mock(side_effect=[all_branches_string]))

    result = manager_fixture.find_original_branch()
    assert result == expected_branch


@patch.object(BranchManager, "commit_and_push_to_branch")
@patch.object(Repo, "untracked_files", ["some-files"])
@patch.object(Repo, "index", Mock())
def test_run_branch_manager(mock_commit_push_method, manager_fixture):
    """
    BranchManager.commit_and_push_to_branch is called when
    the branch has untracked files.
    """
    new_branch = "active-branch-name"
    mock_commit_push_method.return_value = Mock()

    manager_fixture.run_branch_manager(new_branch)
    mock_commit_push_method.assert_called_once()


@patch.object(BranchManager, "commit_and_push_to_branch")
@patch.object(Repo, "untracked_files", [])
@patch.object(Repo, "index")
def test_run_branch_manager(mock_repo_index, mock_commit_push_method, manager_fixture):
    """
    BranchManager.commit_and_push_to_branch is not called when
    the branch is clean of changes.
    """
    new_branch = "active-branch-name"
    mock_commit_push_method.return_value = Mock()
    mock_repo_index.return_value = Mock()
    mock_repo_index.diff = Mock(side_effect=[False])

    manager_fixture.run_branch_manager(new_branch)
    mock_commit_push_method.assert_not_called()
