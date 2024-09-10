#!/usr/bin/env python3
import subprocess
import traceback
import sys

def main():
    try:
        output = add_web_admin()
        print(f"Success. User 'webadmin' is added")
    except:
        traceback.print_exc()
        sys.exit(1)




def add_web_admin():
    cmd = ["sudo", "useradd", "-G", "www-data", "-G", "sudo", "-m", "webadmin"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = p.communicate()
    output = output.strip().decode("utf-8")
    error = error.decode("utf-8")
    if p.returncode != 0:
        print(f"E: {error}")
        raise
    return output



if __name__ == "__main__":
    main()