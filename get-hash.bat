@echo off
setlocal enabledelayedexpansion

for /r %%f in (*.kt *.kts) do (
    certutil -hashfile "%%f" MD5 | find /i /v "hash" | find /i /v "certutil"
)
pause