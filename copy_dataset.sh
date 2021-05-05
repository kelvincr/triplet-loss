dataset="s3://datasets-maestria-2021/"

target=split_3_2_95
tmp_dir=~/tmp/

tmp_target="${tmp_dir}${target}/"

run_ds=~/data/PLANTCLEF/
photo_split="${tmp_dir}photo_split.tar"

if [[ ! -e "$tmp_target" ]]; then
    mkdir "$tmp_target"
fi

echo sync "${tmp_target}"
aws s3 sync "${dataset}${target}/" "${tmp_target}"

if [[ ! -e "$photo_split" ]]; then
    aws s3 cp "${dataset}photo_split.tar" "${tmp_dir}photo_split.tar"
fi

if [[ ! -e "$run_ds" ]]; then
    mkdir --parents "${run_ds}/herbarium/"
    mkdir --parents "${run_ds}/photo/"
else 
    rm -rf "$run_ds"
    mkdir --parents "${run_ds}/herbarium/"
    mkdir --parents "${run_ds}/photo/"
fi

cd "${tmp_target}"

ls

echo "extracting ${photo_split}"
tar --strip-components=2 -C "${run_ds}photo" -xf "${photo_split}"
echo "extracting dataset to ${run_ds}herbarium"
for f in *.tar; do tar  -C "${run_ds}herbarium" -xf "$f";echo "    extracted $f"; done
# for f in *.tar; do tar --strip-components=1 -C "${run_ds}/herbarium" -xf "$f";echo "    extracted $f"; done
rm -rf "${run_ds}photo/108335"
find "$run_ds" -type f | wc -l
