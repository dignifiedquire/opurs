#!/usr/bin/env bash
set -euo pipefail

: "${OPUS_OPUSHD_TEST_VECTORS_URL:?OPUS_OPUSHD_TEST_VECTORS_URL is required}"
: "${OPUS_DRED_TEST_VECTORS_URL:?OPUS_DRED_TEST_VECTORS_URL is required}"

need_opushd=0
need_dred=0

if ! compgen -G "opus_newvectors/qext_vector[0-9][0-9].bit" > /dev/null; then
  need_opushd=1
fi
if ! compgen -G "opus_newvectors/qext_vector[0-9][0-9]fuzz.bit" > /dev/null; then
  need_opushd=1
fi
if ! compgen -G "opus_newvectors/vector*_opus.bit" > /dev/null; then
  need_dred=1
fi
if ! compgen -G "opus_newvectors/vector*_orig.sw" > /dev/null; then
  need_dred=1
fi

if [ "$need_opushd" -eq 1 ]; then
  if [ ! -f opushd_testvectors.tar.gz ]; then
    curl -L "$OPUS_OPUSHD_TEST_VECTORS_URL" -o opushd_testvectors.tar.gz
  fi
  rm -rf opushd_tmp
  mkdir -p opushd_tmp
  tar -xzf opushd_testvectors.tar.gz -C opushd_tmp
  find opushd_tmp -type f -name 'qext_vector*.bit' -exec cp -f {} opus_newvectors/ \;
  rm -rf opushd_tmp
fi

if [ "$need_dred" -eq 1 ]; then
  if [ ! -f dred_testvectors.tar.gz ]; then
    curl -L "$OPUS_DRED_TEST_VECTORS_URL" -o dred_testvectors.tar.gz
  fi
  rm -rf dred_tmp
  mkdir -p dred_tmp
  tar -xzf dred_testvectors.tar.gz -C dred_tmp
  find dred_tmp -type f -name 'vector*_opus.bit' -exec cp -f {} opus_newvectors/ \;
  find dred_tmp -type f -name 'vector*_orig.sw' -exec cp -f {} opus_newvectors/ \;
  rm -rf dred_tmp
fi

compgen -G "opus_newvectors/qext_vector[0-9][0-9].bit" > /dev/null
compgen -G "opus_newvectors/qext_vector[0-9][0-9]fuzz.bit" > /dev/null
compgen -G "opus_newvectors/vector*_opus.bit" > /dev/null
compgen -G "opus_newvectors/vector*_orig.sw" > /dev/null
