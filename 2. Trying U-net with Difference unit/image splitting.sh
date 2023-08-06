split-image ./test/A/ 4 4 -s --output-dir ./images_split/A/ --quiet
rm -f ls ./images_split/A/*_squared.png

split-image ./test/B/ 4 4 -s --output-dir ./images_split/B/ --quiet
rm -f ls ./images_split/B/*_squared.png

split-image ./test/label/ 4 4 -s --output-dir ./images_split/label/ --quiet
rm -f ls ./images_split/label/*_squared.png
