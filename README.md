# Osu Neural Network Created Using Pytorch

- DISCLAIMER : I am not responsible for any consequences that stem from the illicit use of the contents of this repository.

## Info

- Data extracted from replays produced by a slightly modified version of [danser-go](https://github.com/Wieku/danser-go)

- There are two seperate models, one for clicking and one for aiming.

- All skins used are located in [assets/skins](assets/skins)

- Showcase videos [Clicking](https://www.youtube.com/watch?v=ZgHyN98iR1M&t=5s) and [Aiming](https://www.youtube.com/watch?v=YEoSrtow8Qw).

- This repo only works on windows due to how screenshots are taken during testing (Note this is being worked on).

## Quick Start
### Note this process is currently being worked on and may not currently work
- The following assume that your monitor/screen size is 1920x1080 (plan to make it dynamic later)
- Clone this repository and also these forked repositories [osu-framework](https://github.com/TareHimself/osu-framework) and [osu lazer](https://github.com/TareHimself/osu-ml). make sure that osu-framework and osu lazer are in the same directory
- Open up the cloned osu lazer repo and replace the first line in "config.txt" with the path to your cloned version of this repo.
- Open up this repo and install the requirements along with the [wheel](assets/pywin32-228-cp39-cp39-win_amd64.whl).
- Run osu by navigating to the cloned osu lazer repo and running "dotnet run --project osu.Desktop". (You must have DotNet installed for osu to work)
- Run main.py and follow the steps depending on what you want to do.
- You do not need osu lazer open to train but you need it to test and to collect data.

## Example Autopilot play below on [this map](https://osu.ppy.sh/beatmapsets/765778#osu/1627148). The model was trained using [this map](https://osu.ppy.sh/beatmapsets/1721048#osu/3560542).

![goodplay](assets/good-play-autopilot.png)

## Example Relax play below on [this map](https://osu.ppy.sh/beatmapsets/1357624#osu/2809623). The model was trained using [this map](https://osu.ppy.sh/beatmapsets/1511778#osu/3287118).

![goodplay](assets/good-play-relax.png)
