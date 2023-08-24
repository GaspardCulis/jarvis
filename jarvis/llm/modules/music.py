from jarvis.llm.modules.module import LLMModule
from ytmusicapi import YTMusic
import os
import json
import subprocess

class MusicSearch(LLMModule):
    MAX_SEARCH_RESULTS = 3

    def __init__(self) -> None:
        super().__init__("music-search", "Search musics based on a text query, provides the YouTube videoId for each result", {
            "query":{
                "type":"string", 
                "description": "The search query"}
        })
        self.ytmusic = YTMusic(auth=os.getenv("YTMUSICAPI_OAUTH_PATH") or None)  # type: ignore

    def activate(self, arguments: dict) -> str:
        query = arguments.get("query")
        if not query:
            return "Error: No 'query' argument provided"
        search_results = self.ytmusic.search(query, limit=MusicSearch.MAX_SEARCH_RESULTS)
        while len(search_results) > MusicSearch.MAX_SEARCH_RESULTS:
            search_results.pop()
        out = []
        for r in search_results:
            if not r.get("title"): continue
            artists = []
            for a in r.get("artists") or []:
                artists.append(a["name"])
            out.append({
                "title": r.get("title"),
                "artist": ", ".join(artists),
                "duration": r.get("duration"),
                "views": r.get("views"),
                "video-id": r.get("videoId")
            })

        return json.dumps(out)
    
    def get_preprompts(self) -> list[dict]:
        temp_str = self.activate({
            "query": "Energy Drink"
        })
        temp = json.loads(temp_str)
        return [
            {
                "role": "user", 
                "content": "Start the Energy Drink music"
            },
            {
                "role": 
                "assistant", 
                "content": None,
                "function_call": {
                    "name": self.name,
                    "arguments": "{\n  \"query\": \"Energy Drink\"\n}"
                }
            },
            {
                'role': 'function', 
                'name': self.name, 
                'content': temp_str
            },
            {
                "role": "assistant",
                "content": f"The one by {temp[0]['artist']} ?"
            },
            {
                "role": "user", 
                "content": "Yes."
            },
            {
                "role": 
                "assistant", 
                "content": None,
                "function_call": {
                    "name": "music-play",
                    "arguments": "{\n  \"video-id\": \"" + temp[0]["video-id"] + "\"\n}"
                }
            }
        ]

    
    
class MusicPlay(LLMModule):
    def __init__(self) -> None:
        super().__init__("music-play", "Plays music given a YouTube videoId", {
            "video-id": {
                "type":"string", 
                "description":"The YouTube videoId of the song to play"
            }
        })

    def activate(self, arguments: dict) -> str:
        videoId = arguments.get("video-id")
        if not videoId:
            return "Error: No 'video-id' argument provided"
        try:
            # TODO Less system specific
            subprocess.run(f"xfce4-terminal -T jarvis-playback --command=\"mpv --no-video https://youtu.be/{videoId}\"", shell=True)
        except KeyboardInterrupt:
            pass
        return ""

    def get_preprompts(self) -> list[dict]:
        return []
    