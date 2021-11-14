[Done]
1. S3DIS dataset을 ply 파일로 변환하기 (대부분의 previous work에서 처리함)
2. 3D-unet 준비 완료

[To-do-list]

1. Dataloader 만들기[ply 파일에서 읽어오는..]
2. 1번이 완료되면, training code 만들기.
3. 3D-unet에 sparse convolution layer 적용시키기 [https://github.com/traveller59/spconv 참조.]
4. Visualization code 만들기
5. ...

[How to prepare dataset?]
1. Download S3DIS dataset in [https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1]
(form 아무렇게나 작성하면 링크 나옴)

2. Prepare 'Stanford3dDataset_v1.2_Aligned_Version.zip', and extract zip file to './data'
3. Compile given cpp files for preprocessing by 'sh compile_op.sh'
3. do [python utils/s3dis_dp.py]
4. Done.

