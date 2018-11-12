#!/bin/bash
# this script launches the Virtual Box Vms

# parse command line args
if [ "$1" = "" -o "$1" = "start" ] ; then
    if [ "$2" = "-h" ] ; then
        echo "Starting VMs -- headless"
        # launch VMs with no GUI
        VBoxManage startvm "Metasploitable-2" --type headless
    else
        echo "Starting VMs -- GUI"
        # launch VMs with GUI
        VBoxManage startvm "Metasploitable-2"
    exit 0
    fi
elif [ "$1" = "close" ] ; then
    echo "Closing VMs"
    VBoxManage controlvm "Metasploitable-2" poweroff
    exit 0
else
    echo "Usage: setup.sh [start || close]"
    exit 1
fi
