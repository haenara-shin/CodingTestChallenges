import youtube_dl

with youtube_dl.YoutubeDL() as ytl:
    ytl.download(['https://youtu.be/tDukIfFzX18'])

