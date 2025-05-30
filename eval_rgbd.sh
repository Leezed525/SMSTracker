# eval depthtrack
cd Depthtrack_workspace
vot evaluate  SMSTracker
vot analysis  SMSTracker
vot report SMSTracker
cd ..

## eval vot22-rgbd
#cd VOT22RGBD_workspace
#vot evaluate  SMSTracker
#vot analysis  SMSTracker
#vot report SMSTracker
#cd ..
#
