# buiding exe for test_system.py (cannot use --onefile or it will be mistreated as a trogan)
# ref: https://stackoverflow.com/questions/43777106/program-made-with-pyinstaller-now-seen-as-a-trojan-horse-by-avg


pyinstaller --noconsole --hidden-import  charset_normalizer.md__mypyc --add-data "res/*.png;." --add-data "res/*.svg;."  -i "res/app.ico" -n "System Test Tool v0.1" test_system.py