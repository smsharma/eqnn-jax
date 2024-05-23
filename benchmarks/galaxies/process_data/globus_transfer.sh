#!/bin/bash

SOURCE_ENDPOINT="e0eae0aa-5bca-11ea-9683-0e56c063f437"
DESTINATION_ENDPOINT="YOUR DESTINATION ENDPOINT" 
SOURCE_PATH="/Halos/Rockstar/BSQ"
DESTINATION_PATH="YOUR DESTINATION PATH"

TRANSFER_LIST="transfer_list.txt"

> $TRANSFER_LIST

# List directories in the source path
dirs=$(globus ls $SOURCE_ENDPOINT:$SOURCE_PATH) 
dir_count=$(echo "$dirs" | wc -l)
# Print the count of directories
echo "Number of directories found in $SOURCE_PATH: $dir_count"

# Loop over each directory found
for i in $dirs; do
    echo "$SOURCE_PATH/$i/out_10.list $DESTINATION_PATH/$i/out_10.list" >> $TRANSFER_LIST
done
# Ensure the transfer list is properly formatted
echo "Starting transfer using transfer list..."
globus transfer $SOURCE_ENDPOINT $DESTINATION_ENDPOINT --batch $TRANSFER_LIST