import pytest

from configparser import NoSectionError
from unittest.mock import patch, Mock, call
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


@patch.object(Repo, "remotes", [])
@patch.object(Repo, "create_remote")
@patch.object(Repo, "delete_remote")
def test_create_remote_with_token_remote_does_not_exist(
    mock_delete_remote, mock_create_remote, manager_fixture
):
    """
    BranchManager.create_remote_with_token() does not call Repo.delete_remote,
    because the remote we want to create does not already exist.
    Then, it creates the new remote with the give user and token.
    """
    manager_fixture.create_remote_with_token()

    mock_delete_remote.assert_not_called()
    mock_create_remote.assert_called_with(
        "new_origin", "https://me:abc@gitlab.com/ska-telescope/external/rascil.git"
    )


@patch.object(Repo, "remotes", ["new_origin"])
@patch.object(Repo, "create_remote")
@patch.object(Repo, "delete_remote")
def test_create_remote_with_token_remote_exists(
    mock_delete_remote, mock_create_remote, manager_fixture
):
    """
    BranchManager.create_remote_with_token() calls Repo.delete_remote,
    because the remote ("new_origin") we want to create already exists.
    After deleting, it recreates the remote with the provided user and token.
    """
    manager_fixture.create_remote_with_token()

    mock_delete_remote.assert_called()
    mock_create_remote.assert_called_with(
        "new_origin", "https://me:abc@gitlab.com/ska-telescope/external/rascil.git"
    )


@patch.object(BranchManager, "set_git_config", Mock())
@patch.object(Repo, "heads", [])
@patch.object(Repo, "git")
def test_commit_and_push_to_branch(mock_git, manager_fixture):
    """
    Given
        BranchManager.commit_and_push_to_branch
    When
        - the new branch name is not in repo.heads
        - custom commit message is not provided
    Then
        - Repo.delete_head is not called
        - the commit message is the default message (see method)

    """
    new_branch = "my-new-branch"
    mock_git.checkout = Mock()

    manager_fixture.commit_and_push_to_branch(new_branch)