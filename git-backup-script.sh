#!/bin/bash

timestamp() {
  date +"%c"
}

cd /data/yiph2/UMLS-DL
/usr/bin/git add -A
/usr/bin/git commit -m "Latest Update"
/usr/bin/git push -u origin master
timestamp >> git-monitor.txt
