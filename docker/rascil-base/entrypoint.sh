#!/bin/bash
# Create a local user on the fly using the LOCAL_USER_ID environment variable.
USER_ID=${LOCAL_USER_ID:-9001}
useradd -s /bin/bash -u $USER_ID -o -m rascil
export HOME=/home/rascil
exec gosu rascil "$@"