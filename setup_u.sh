#!/bin/bash

# See this stackoverflow question
# http://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within
# for the magic in this command
SETUP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#
# Base package root. All the other releavant folders are relative to this
# location.
#
export SOURCE_ROOT=$SETUP_DIR
echo "SOURCE_ROOT set to " $SOURCE_ROOT

#
# Add the root folder to the $PYTHONPATH so that we can effectively import
# the relevant modules.
#
# Check if source root is already added to PYTHONPATH before writing 
# (if yes, it doesn't make anything)
#
if grep -R 'export PYTHONPATH="${PYTHONPATH}:'${SOURCE_ROOT}'"' ~/.bashrc
then
	echo 'Root already added'
else
	echo 'export PYTHONPATH="${PYTHONPATH}:'${SOURCE_ROOT}'"' >> ~/.bashrc
	echo $SOURCE_ROOT' successfully added to $PYTHONPATH'
fi
