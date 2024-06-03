## How to generate exe:

pyinstaller --noconsole --onefile --hidden-import  charset_normalizer.md__mypyc --add-data "res/*.png;." --add-data "res/*.svg;."  -i "res/app.ico" -n "ISD_UI_Mockup_v0_0_2" Camera_view.py 

pyinstaller --hidden-import  charset_normalizer.md__mypyc --add-data "res/*.png;." --add-data "res/*.svg;."  -i "res/app_100_100.ico" -n "test.exe" Camera_view.py 