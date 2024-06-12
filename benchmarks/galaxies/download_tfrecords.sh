#!/bin/bash

zip_url='https://zenodo.org/api/records/11479419/files-archive'

output_dir="/tmp"
zip_file="$output_dir/quijote_records.zip"
unzip_dir="/quijote_records"

mkdir -p "$output_dir"
mkdir -p "$unzip_dir"

echo "Downloading $zip_url..."
curl -L -o "$zip_file" "$zip_url"

echo "Unzipping $zip_file to $unzip_dir..."
unzip "$zip_file" -d "$unzip_dir"
rm -rf "$output_dir"

echo "Download complete."