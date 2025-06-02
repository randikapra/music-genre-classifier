# # ## --version 1.0-- ##
# # #!/usr/bin/env python
# # # coding: utf-8

# # """
# # Student Dataset Generator Script

# # This script helps generate a dataset for an additional genre not in the original dataset.
# # You can use it to create your student_dataset.csv file.

# # There are two approaches:
# # 1. Manual entry (safer): Enter song information manually
# # 2. Automated API collection (requires API key): Use LyricsGenius API

# # Note: If you use the API approach, make sure you have appropriate permissions to use the data.
# # """

# # import os
# # import csv
# # import time
# # import re
# # import pandas as pd
# # from datetime import datetime

# # try:
# #     import lyricsgenius
# #     GENIUS_AVAILABLE = True
# # except ImportError:
# #     GENIUS_AVAILABLE = False

# # # Constants
# # OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "student_dataset.csv")
# # MIN_SONGS = 10

# # def clean_lyrics(lyrics):
# #     """Clean lyrics text"""
# #     if not lyrics:
# #         return ""
    
# #     # Remove [Verse], [Chorus], etc.
# #     lyrics = re.sub(r'\[.*?\]', '', lyrics)
    
# #     # Remove extra whitespace
# #     lyrics = re.sub(r'\s+', ' ', lyrics).strip()
    
# #     return lyrics

# # def manual_entry():
# #     """Manually enter song information"""
# #     print("\n=== Manual Song Entry ===")
    
# #     # Create data directory if it doesn't exist
# #     os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
# #     # Get genre name
# #     genre = input("Enter the genre name for all songs: ").strip().lower()
# #     print(f"Genre set to: {genre}")
# #     print(f"You need to enter at least {MIN_SONGS} songs. Press Ctrl+C to stop at any time.")
    
# #     # Check if file exists and load existing data
# #     songs = []
# #     if os.path.exists(OUTPUT_FILE):
# #         try:
# #             df = pd.read_csv(OUTPUT_FILE)
# #             songs = df.to_dict('records')
# #             print(f"Loaded {len(songs)} existing songs from {OUTPUT_FILE}")
# #         except Exception as e:
# #             print(f"Error loading existing data: {e}")
    
# #     # Start entry
# #     count = len(songs)
# #     try:
# #         while count < MIN_SONGS:
# #             print(f"\nSong {count+1}/{MIN_SONGS}")
# #             artist = input("Artist name: ").strip()
# #             track = input("Track name: ").strip()
            
# #             while True:
# #                 year = input("Release year (YYYY): ").strip()
# #                 if re.match(r'^\d{4}$', year):
# #                     break
# #                 print("Invalid year format. Please enter a 4-digit year.")
            
# #             print("Enter lyrics (type 'END' on a new line when finished):")
# #             lyrics_lines = []
# #             while True:
# #                 line = input()
# #                 if line.strip() == "END":
# #                     break
# #                 lyrics_lines.append(line)
            
# #             lyrics = "\n".join(lyrics_lines)
# #             lyrics = clean_lyrics(lyrics)
            
# #             songs.append({
# #                 "artist_name": artist,
# #                 "track_name": track,
# #                 "release_date": int(year),
# #                 "genre": genre,
# #                 "lyrics": lyrics
# #             })
            
# #             # Save after each song
# #             df = pd.DataFrame(songs)
# #             df.to_csv(OUTPUT_FILE, index=False)
# #             print(f"Song added. Progress: {len(songs)}/{MIN_SONGS}")
            
# #             count += 1
    
# #     except KeyboardInterrupt:
# #         print("\nEntry stopped by user.")
    
# #     # Final save
# #     df = pd.DataFrame(songs)
# #     df.to_csv(OUTPUT_FILE, index=False)
    
# #     print(f"\nTotal songs collected: {len(songs)}")
# #     print(f"Data saved to {OUTPUT_FILE}")

# # def api_collection():
# #     """Collect data using LyricsGenius API"""
# #     if not GENIUS_AVAILABLE:
# #         print("LyricsGenius package not installed. Install it with:")
# #         print("pip install lyricsgenius")
# #         return
    
# #     print("\n=== Genius API Song Collection ===")
    
# #     # Create data directory if it doesn't exist
# #     os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
# #     # Get API key
# #     api_key = input("Enter your Genius API key: ").strip()
    
# #     # Initialize Genius API
# #     try:
# #         genius = lyricsgenius.Genius(api_key)
# #         genius.verbose = False
# #     except Exception as e:
# #         print(f"Error initializing Genius API: {e}")
# #         return
    
# #     # Get genre name
# #     genre = input("Enter the genre name for all songs: ").strip().lower()
# #     print(f"Genre set to: {genre}")
    
# #     # Get artists
# #     print("Enter artist names known for this genre (one per line, type 'END' when finished):")
# #     artists = []
# #     while True:
# #         artist = input().strip()
# #         if artist == "END":
# #             break
# #         artists.append(artist)
    
# #     if not artists:
# #         print("No artists entered. Exiting.")
# #         return
    
# #     print(f"Collecting songs for {len(artists)} artists in the {genre} genre.")
    
# #     # Check if file exists and load existing data
# #     songs = []
# #     if os.path.exists(OUTPUT_FILE):
# #         try:
# #             df = pd.read_csv(OUTPUT_FILE)
# #             songs = df.to_dict('records')
# #             print(f"Loaded {len(songs)} existing songs from {OUTPUT_FILE}")
# #         except Exception as e:
# #             print(f"Error loading existing data: {e}")
    
# #     # Collect songs
# #     for artist_name in artists:
# #         print(f"\nSearching for songs by {artist_name}...")
        
# #         try:
# #             # Get artist
# #             artist = genius.search_artist(artist_name, max_songs=30, sort="popularity")
            
# #             if not artist:
# #                 print(f"Could not find artist: {artist_name}")
# #                 continue
            
# #             # Process songs
# #             for song in artist.songs:
# #                 if len(songs) >= MIN_SONGS:
# #                     break
                
# #                 # Skip if song has no lyrics
# #                 if not song.lyrics:
# #                     continue
                
# #                 # Get release year
# #                 try:
# #                     year_str = song.year
# #                     if year_str and re.match(r'^\d{4}$', year_str):
# #                         year = int(year_str)
# #                     else:
# #                         # Use current year as fallback
# #                         year = datetime.now().year
# #                 except:
# #                     year = datetime.now().year
                
# #                 # Clean lyrics
# #                 lyrics = clean_lyrics(song.lyrics)
                
# #                 # Add song to collection
# #                 songs.append({
# #                     "artist_name": artist_name,
# #                     "track_name": song.title,
# #                     "release_date": year,
# #                     "genre": genre,
# #                     "lyrics": lyrics
# #                 })
                
# #                 print(f"Added: {song.title} - Progress: {len(songs)}/{MIN_SONGS}")
                
# #                 # Save after each song
# #                 df = pd.DataFrame(songs)
# #                 df.to_csv(OUTPUT_FILE, index=False)
                
# #                 # Respect API rate limits
# #                 time.sleep(1)
        
# #         except Exception as e:
# #             print(f"Error processing artist {artist_name}: {e}")
    
# #     # Final report
# #     print(f"\nTotal songs collected: {len(songs)}")
# #     print(f"Data saved to {OUTPUT_FILE}")
    
# #     if len(songs) < MIN_SONGS:
# #         print(f"Warning: Only collected {len(songs)} songs, which is less than the minimum required ({MIN_SONGS}).")
# #         print("Consider adding more artists or using manual entry to reach the minimum.")

# # def merge_datasets():
# #     """Merge student dataset with the Mendeley dataset"""
# #     student_file = OUTPUT_FILE
# #     mendeley_file = os.path.join(os.path.dirname(OUTPUT_FILE), "mendeley_dataset.csv")
# #     merged_file = os.path.join(os.path.dirname(OUTPUT_FILE), "merged_dataset.csv")
    
# #     if not os.path.exists(student_file):
# #         print(f"Error: Student dataset not found at {student_file}")
# #         return
    
# #     if not os.path.exists(mendeley_file):
# #         print(f"Error: Mendeley dataset not found at {mendeley_file}")
# #         return
    
# #     try:
# #         # Load datasets
# #         student_df = pd.read_csv(student_file)
# #         mendeley_df = pd.read_csv(mendeley_file)
        
# #         # Select required columns from Mendeley dataset
# #         mendeley_df = mendeley_df[["artist_name", "track_name", "release_date", "genre", "lyrics"]]
        
# #         # Concatenate datasets
# #         merged_df = pd.concat([mendeley_df, student_df], ignore_index=True)
        
# #         # Save merged dataset
# #         merged_df.to_csv(merged_file, index=False)
        
# #         print(f"Successfully merged datasets:")
# #         print(f"- Mendeley dataset: {len(mendeley_df)} songs")
# #         print(f"- Student dataset: {len(student_df)} songs")
# #         print(f"- Merged dataset: {len(merged_df)} songs")
# #         print(f"Merged dataset saved to {merged_file}")
    
# #     except Exception as e:
# #         print(f"Error merging datasets: {e}")

# # def main():
# #     """Main function to run the script"""
# #     print("\n=== Student Dataset Generator ===")
# #     print("This script helps you create a dataset for an additional music genre.")
    
# #     while True:
# #         print("\nChoose an option:")
# #         print("1. Manual song entry")
# #         print("2. Collect songs using Genius API")
# #         print("3. Merge with Mendeley dataset")
# #         print("4. Exit")
        
# #         choice = input("Enter your choice (1-4): ").strip()
        
# #         if choice == "1":
# #             manual_entry()
# #         elif choice == "2":
# #             api_collection()
# #         elif choice == "3":
# #             merge_datasets()
# #         elif choice == "4":
# #             print("Exiting. Goodbye!")
# #             break
# #         else:
# #             print("Invalid choice. Please enter a number between 1 and 4.")

# # if __name__ == "__main__":
# #     # Q3XZOS7nlFx-3nLwM1-uf_Jr68XSQx7DggOc43F_QcK2aOC0qxjTXQxycnYtU6Do
# #     api_client_ID='Q3XZOS7nlFx-3nLwM1-uf_Jr68XSQx7DggOc43F_QcK2aOC0qxjTXQxycnYtU6Do'
# #     api_client_Token='e9bk2MUfv_VHRBJsCVamROkB9bFtn_oTOb6-zI4tgzUNlmBcax5m2phWd0TOOpAY'
# #     main()


# # ## --version 2.0-- ##
# # # import os
# # # import re
# # # import time
# # # import pandas as pd
# # # from datetime import datetime
# # # import configparser
# # # from pathlib import Path

# # # # Check if lyricsgenius is available
# # # try:
# # #     import lyricsgenius
# # #     GENIUS_AVAILABLE = True
# # # except ImportError:
# # #     GENIUS_AVAILABLE = False

# # # # Constants
# # # DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "data"
# # # OUTPUT_FILE = DATA_DIR / "student_dataset.csv"
# # # MENDELEY_FILE = DATA_DIR / "mendeley_dataset.csv"
# # # MERGED_FILE = DATA_DIR / "merged_dataset.csv"
# # # CONFIG_FILE = Path(os.path.dirname(os.path.abspath(__file__))) / "config.ini"
# # # MIN_SONGS = 10

# # # def load_config():
# # #     """Load configuration from config file or create one if it doesn't exist"""
# # #     config = configparser.ConfigParser()
    
# # #     if os.path.exists(CONFIG_FILE):
# # #         config.read(CONFIG_FILE)
# # #     else:
# # #         config['API'] = {
# # #             'genius_client_id': '',
# # #             'genius_client_token': ''
# # #         }
# # #         os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
# # #         with open(CONFIG_FILE, 'w') as f:
# # #             config.write(f)
    
# # #     return config

# # # def save_config(config):
# # #     """Save configuration to config file"""
# # #     with open(CONFIG_FILE, 'w') as f:
# # #         config.write(f)

# # # def clean_lyrics(lyrics):
# # #     """Clean lyrics text more thoroughly"""
# # #     if not lyrics:
# # #         return ""
    
# # #     # Remove [Verse], [Chorus], etc.
# # #     lyrics = re.sub(r'\[.*?\]', '', lyrics)
    
# # #     # Remove line numbers and timestamps
# # #     lyrics = re.sub(r'^\d+\s*', '', lyrics, flags=re.MULTILINE)
# # #     lyrics = re.sub(r'\d+:\d+', '', lyrics)
    
# # #     # Remove HTML/XML-like tags
# # #     lyrics = re.sub(r'<.*?>', '', lyrics)
    
# # #     # Remove "ContributorsTranslationsRomanization" and similar metadata
# # #     lyrics = re.sub(r'\d+\s*Contributors\w*\s*Translations\w*\s*Romanization\w*', '', lyrics)
# # #     lyrics = re.sub(r'Lyrics\s*', '', lyrics)
    
# # #     # Remove extra whitespace
# # #     lyrics = re.sub(r'\s+', ' ', lyrics).strip()
    
# # #     return lyrics

# # # def compute_analytics(lyrics):
# # #     """
# # #     Compute basic analytics for a song to match the Mendeley dataset structure
# # #     This is a simplified version - in a real implementation, you'd use libraries 
# # #     like librosa or spotipy to get actual audio features
# # #     """
# # #     # Create a dictionary with default values
# # #     analytics = {
# # #         'len': len(lyrics.split()),  # Word count as a simple length metric
# # #         'dating': 0.0,
# # #         'violence': 0.0,
# # #         'world/life': 0.0,
# # #         'night/time': 0.0,
# # #         'shake the audience': 0.0,
# # #         'family/gospel': 0.0,
# # #         'romantic': 0.0,
# # #         'communication': 0.0,
# # #         'obscene': 0.0,
# # #         'music': 0.0,
# # #         'movement/places': 0.0,
# # #         'light/visual perceptions': 0.0,
# # #         'family/spiritual': 0.0,
# # #         'like/girls': 0.0,
# # #         'sadness': 0.0,
# # #         'feelings': 0.0,
# # #         'danceability': 0.0,
# # #         'loudness': 0.0,
# # #         'acousticness': 0.0,
# # #         'instrumentalness': 0.0,
# # #         'valence': 0.0,
# # #         'energy': 0.0,
# # #         'topic': 'unknown',
# # #         'age': 1.0
# # #     }
    
# # #     # In a real implementation, you'd calculate these values using NLP or audio analysis
# # #     # For now, we're just setting some placeholder values
    
# # #     # Example: Set the "romantic" score based on keyword matching
# # #     romantic_words = ['love', 'heart', 'romance', 'kiss', 'embrace', 'passion', 'प्यार', 'दिल', 'इश्क़']
# # #     romantic_count = sum(1 for word in romantic_words if word.lower() in lyrics.lower())
# # #     analytics['romantic'] = min(1.0, romantic_count / 10)
    
# # #     # Example: Set the "sadness" score based on keyword matching
# # #     sad_words = ['sad', 'cry', 'tear', 'pain', 'hurt', 'goodbye', 'दर्द', 'जुदाई', 'अलविदा']
# # #     sad_count = sum(1 for word in sad_words if word.lower() in lyrics.lower())
# # #     analytics['sadness'] = min(1.0, sad_count / 10)
    
# # #     # Determine the most prevalent topic based on the highest score
# # #     if analytics['romantic'] > analytics['sadness']:
# # #         analytics['topic'] = 'romantic'
# # #     else:
# # #         analytics['topic'] = 'sadness'
    
# # #     return analytics

# # # def manual_entry():
# # #     """Manually enter song information"""
# # #     print("\n=== Manual Song Entry ===")
    
# # #     # Create data directory if it doesn't exist
# # #     os.makedirs(DATA_DIR, exist_ok=True)
    
# # #     # Get genre name
# # #     genre = input("Enter the genre name for all songs: ").strip().lower()
# # #     print(f"Genre set to: {genre}")
# # #     print(f"You need to enter at least {MIN_SONGS} songs. Press Ctrl+C to stop at any time.")
    
# # #     # Check if file exists and load existing data
# # #     songs = []
# # #     if os.path.exists(OUTPUT_FILE):
# # #         try:
# # #             df = pd.read_csv(OUTPUT_FILE)
# # #             songs = df.to_dict('records')
# # #             print(f"Loaded {len(songs)} existing songs from {OUTPUT_FILE}")
# # #         except Exception as e:
# # #             print(f"Error loading existing data: {e}")
    
# # #     # Start entry
# # #     count = len(songs)
# # #     try:
# # #         while count < MIN_SONGS:
# # #             print(f"\nSong {count+1}/{MIN_SONGS}")
# # #             artist = input("Artist name: ").strip()
# # #             track = input("Track name: ").strip()
            
# # #             while True:
# # #                 year = input("Release year (YYYY): ").strip()
# # #                 if re.match(r'^\d{4}$', year):
# # #                     break
# # #                 print("Invalid year format. Please enter a 4-digit year.")
            
# # #             print("Enter lyrics (type 'END' on a new line when finished):")
# # #             lyrics_lines = []
# # #             while True:
# # #                 line = input()
# # #                 if line.strip() == "END":
# # #                     break
# # #                 lyrics_lines.append(line)
            
# # #             lyrics = "\n".join(lyrics_lines)
# # #             lyrics = clean_lyrics(lyrics)
            
# # #             # Create base song entry
# # #             song_entry = {
# # #                 "artist_name": artist,
# # #                 "track_name": track,
# # #                 "release_date": int(year),
# # #                 "genre": genre,
# # #                 "lyrics": lyrics
# # #             }
            
# # #             # Add analytics to match Mendeley dataset
# # #             analytics = compute_analytics(lyrics)
# # #             song_entry.update(analytics)
            
# # #             songs.append(song_entry)
            
# # #             # Save after each song
# # #             df = pd.DataFrame(songs)
# # #             df.to_csv(OUTPUT_FILE, index=False)
# # #             print(f"Song added. Progress: {len(songs)}/{MIN_SONGS}")
            
# # #             count += 1
    
# # #     except KeyboardInterrupt:
# # #         print("\nEntry stopped by user.")
    
# # #     # Final save
# # #     df = pd.DataFrame(songs)
# # #     df.to_csv(OUTPUT_FILE, index=False)
    
# # #     print(f"\nTotal songs collected: {len(songs)}")
# # #     print(f"Data saved to {OUTPUT_FILE}")

# # # def api_collection():
# # #     """Collect data using LyricsGenius API"""
# # #     if not GENIUS_AVAILABLE:
# # #         print("LyricsGenius package not installed. Install it with:")
# # #         print("pip install lyricsgenius")
# # #         return
    
# # #     print("\n=== Genius API Song Collection ===")
    
# # #     # Create data directory if it doesn't exist
# # #     os.makedirs(DATA_DIR, exist_ok=True)
    
# # #     # Load config
# # #     config = load_config()
    
# # #     # Get API key
# # #     api_key = input("Enter your Genius API key (press Enter to use saved key): ").strip()
    
# # #     if not api_key and 'API' in config and 'genius_client_token' in config['API']:
# # #         api_key = config['API']['genius_client_token']
# # #         print("Using saved API key.")
    
# # #     if not api_key:
# # #         print("No API key provided. Exiting.")
# # #         return
    
# # #     # Save API key for future use
# # #     if 'API' not in config:
# # #         config['API'] = {}
# # #     config['API']['genius_client_token'] = api_key
# # #     save_config(config)
    
# # #     # Initialize Genius API
# # #     try:
# # #         genius = lyricsgenius.Genius(api_key)
# # #         genius.verbose = False
# # #         # Increase timeout for more reliable fetching
# # #         genius.timeout = 15
# # #         # Remove section headers from lyrics
# # #         genius.remove_section_headers = True
# # #     except Exception as e:
# # #         print(f"Error initializing Genius API: {e}")
# # #         return
    
# # #     # Get genre name
# # #     genre = input("Enter the genre name for all songs: ").strip().lower()
# # #     print(f"Genre set to: {genre}")
    
# # #     # Get artists
# # #     print("Enter artist names known for this genre (one per line, type 'END' when finished):")
# # #     artists = []
# # #     while True:
# # #         artist = input().strip()
# # #         if artist == "END":
# # #             break
# # #         artists.append(artist)
    
# # #     if not artists:
# # #         print("No artists entered. Exiting.")
# # #         return
    
# # #     print(f"Collecting songs for {len(artists)} artists in the {genre} genre.")
    
# # #     # Check if file exists and load existing data
# # #     songs = []
# # #     if os.path.exists(OUTPUT_FILE):
# # #         try:
# # #             df = pd.read_csv(OUTPUT_FILE)
# # #             songs = df.to_dict('records')
# # #             print(f"Loaded {len(songs)} existing songs from {OUTPUT_FILE}")
# # #         except Exception as e:
# # #             print(f"Error loading existing data: {e}")
    
# # #     # Collect songs
# # #     for artist_name in artists:
# # #         print(f"\nSearching for songs by {artist_name}...")
        
# # #         try:
# # #             # Get artist
# # #             artist = genius.search_artist(artist_name, max_songs=30, sort="popularity")
            
# # #             if not artist:
# # #                 print(f"Could not find artist: {artist_name}")
# # #                 continue
            
# # #             # Process songs
# # #             for song in artist.songs:
# # #                 if len(songs) >= MIN_SONGS:
# # #                     break
                
# # #                 # Skip if song has no lyrics
# # #                 if not song.lyrics:
# # #                     continue
                
# # #                 # Get release year
# # #                 try:
# # #                     year_str = song.year
# # #                     if year_str and re.match(r'^\d{4}$', year_str):
# # #                         year = int(year_str)
# # #                     else:
# # #                         # Use current year as fallback
# # #                         year = datetime.now().year
# # #                 except:
# # #                     year = datetime.now().year
                
# # #                 # Clean lyrics
# # #                 lyrics = clean_lyrics(song.lyrics)
                
# # #                 # Create base song entry
# # #                 song_entry = {
# # #                     "artist_name": artist_name,
# # #                     "track_name": song.title,
# # #                     "release_date": year,
# # #                     "genre": genre,
# # #                     "lyrics": lyrics
# # #                 }
                
# # #                 # Add analytics to match Mendeley dataset
# # #                 analytics = compute_analytics(lyrics)
# # #                 song_entry.update(analytics)
                
# # #                 # Add song to collection
# # #                 songs.append(song_entry)
                
# # #                 print(f"Added: {song.title} - Progress: {len(songs)}/{MIN_SONGS}")
                
# # #                 # Save after each song
# # #                 df = pd.DataFrame(songs)
# # #                 df.to_csv(OUTPUT_FILE, index=False)
                
# # #                 # Respect API rate limits
# # #                 time.sleep(1)
        
# # #         except Exception as e:
# # #             print(f"Error processing artist {artist_name}: {e}")
    
# # #     # Final report
# # #     print(f"\nTotal songs collected: {len(songs)}")
# # #     print(f"Data saved to {OUTPUT_FILE}")
    
# # #     if len(songs) < MIN_SONGS:
# # #         print(f"Warning: Only collected {len(songs)} songs, which is less than the minimum required ({MIN_SONGS}).")
# # #         print("Consider adding more artists or using manual entry to reach the minimum.")

# # # def merge_datasets():
# # #     """Merge student dataset with the Mendeley dataset"""
# # #     if not os.path.exists(OUTPUT_FILE):
# # #         print(f"Error: Student dataset not found at {OUTPUT_FILE}")
# # #         return
    
# # #     if not os.path.exists(MENDELEY_FILE):
# # #         print(f"Error: Mendeley dataset not found at {MENDELEY_FILE}")
# # #         return
    
# # #     try:
# # #         # Load datasets
# # #         student_df = pd.read_csv(OUTPUT_FILE)
# # #         mendeley_df = pd.read_csv(MENDELEY_FILE)
        
# # #         # Get column names from Mendeley dataset to ensure consistency
# # #         mendeley_columns = mendeley_df.columns.tolist()
        
# # #         # Check and add missing columns to student dataset
# # #         for col in mendeley_columns:
# # #             if col not in student_df.columns:
# # #                 print(f"Warning: Column '{col}' not found in student dataset. Adding with default values.")
# # #                 # Add default values based on column type
# # #                 if col in ['artist_name', 'track_name', 'genre', 'lyrics', 'topic']:
# # #                     student_df[col] = 'unknown'
# # #                 elif col == 'release_date':
# # #                     student_df[col] = datetime.now().year
# # #                 else:
# # #                     student_df[col] = 0.0
        
# # #         # Ensure columns are in the same order
# # #         student_df = student_df[mendeley_columns]
        
# # #         # Concatenate datasets
# # #         merged_df = pd.concat([mendeley_df, student_df], ignore_index=True)
        
# # #         # Save merged dataset
# # #         merged_df.to_csv(MERGED_FILE, index=False)
        
# # #         print(f"Successfully merged datasets:")
# # #         print(f"- Mendeley dataset: {len(mendeley_df)} songs")
# # #         print(f"- Student dataset: {len(student_df)} songs")
# # #         print(f"- Merged dataset: {len(merged_df)} songs")
# # #         print(f"Merged dataset saved to {MERGED_FILE}")
    
# # #     except Exception as e:
# # #         print(f"Error merging datasets: {e}")

# # # def analyze_dataset():
# # #     """Analyze the dataset and print statistics"""
# # #     print("\n=== Dataset Analysis ===")
    
# # #     if not os.path.exists(OUTPUT_FILE):
# # #         print(f"Error: Student dataset not found at {OUTPUT_FILE}")
# # #         return
    
# # #     try:
# # #         df = pd.read_csv(OUTPUT_FILE)
        
# # #         print(f"Total songs: {len(df)}")
# # #         print(f"Artists: {df['artist_name'].nunique()}")
        
# # #         print("\nGenre distribution:")
# # #         genre_counts = df['genre'].value_counts()
# # #         for genre, count in genre_counts.items():
# # #             print(f"- {genre}: {count} songs")
        
# # #         print("\nRelease year distribution:")
# # #         year_counts = df['release_date'].value_counts().sort_index()
# # #         for year, count in year_counts.items():
# # #             print(f"- {year}: {count} songs")
        
# # #         # Calculate average song length
# # #         if 'len' in df.columns:
# # #             avg_length = df['len'].mean()
# # #             print(f"\nAverage song length: {avg_length:.2f} words")
        
# # #         # If topic column exists, show distribution
# # #         if 'topic' in df.columns:
# # #             print("\nTopic distribution:")
# # #             topic_counts = df['topic'].value_counts()
# # #             for topic, count in topic_counts.items():
# # #                 print(f"- {topic}: {count} songs")
        
# # #     except Exception as e:
# # #         print(f"Error analyzing dataset: {e}")

# # # def validate_dataset():
# # #     """Validate the dataset for completeness and consistency"""
# # #     print("\n=== Dataset Validation ===")
    
# # #     if not os.path.exists(OUTPUT_FILE):
# # #         print(f"Error: Student dataset not found at {OUTPUT_FILE}")
# # #         return
    
# # #     try:
# # #         df = pd.read_csv(OUTPUT_FILE)
        
# # #         # Check for missing values
# # #         missing_values = df.isnull().sum()
# # #         if missing_values.sum() > 0:
# # #             print("Missing values found:")
# # #             for col, count in missing_values.items():
# # #                 if count > 0:
# # #                     print(f"- {col}: {count} missing values")
# # #         else:
# # #             print("No missing values found.")
        
# # #         # Check for empty lyrics
# # #         empty_lyrics = df[df['lyrics'].str.strip() == ''].shape[0]
# # #         if empty_lyrics > 0:
# # #             print(f"Warning: {empty_lyrics} songs have empty lyrics.")
        
# # #         # Check for very short lyrics
# # #         short_lyrics = df[df['lyrics'].str.split().str.len() < 10].shape[0]
# # #         if short_lyrics > 0:
# # #             print(f"Warning: {short_lyrics} songs have very short lyrics (less than 10 words).")
        
# # #         # Check for consistent release dates
# # #         invalid_years = df[(df['release_date'] < 1900) | (df['release_date'] > datetime.now().year + 1)].shape[0]
# # #         if invalid_years > 0:
# # #             print(f"Warning: {invalid_years} songs have potentially invalid release years.")
        
# # #         print("Validation complete.")
    
# # #     except Exception as e:
# # #         print(f"Error validating dataset: {e}")

# # # def main():
# # #     """Main function to run the script"""
# # #     print("\n=== Student Dataset Generator ===")
# # #     print("This script helps you create a dataset for an additional music genre.")
    
# # #     # Create data directory if it doesn't exist
# # #     os.makedirs(DATA_DIR, exist_ok=True)
    
# # #     while True:
# # #         print("\nChoose an option:")
# # #         print("1. Manual song entry")
# # #         print("2. Collect songs using Genius API")
# # #         print("3. Merge with Mendeley dataset")
# # #         print("4. Analyze dataset")
# # #         print("5. Validate dataset")
# # #         print("6. Exit")
        
# # #         choice = input("Enter your choice (1-6): ").strip()
        
# # #         if choice == "1":
# # #             manual_entry()
# # #         elif choice == "2":
# # #             api_collection()
# # #         elif choice == "3":
# # #             merge_datasets()
# # #         elif choice == "4":
# # #             analyze_dataset()
# # #         elif choice == "5":
# # #             validate_dataset()
# # #         elif choice == "6":
# # #             print("Exiting. Goodbye!")
# # #             break
# # #         else:
# # #             print("Invalid choice. Please enter a number between 1 and 6.")
# # # # if __name__ == "__main__":
# # # #     # Q3XZOS7nlFx-3nLwM1-uf_Jr68XSQx7DggOc43F_QcK2aOC0qxjTXQxycnYtU6Do
# # # #     api_client_ID='Q3XZOS7nlFx-3nLwM1-uf_Jr68XSQx7DggOc43F_QcK2aOC0qxjTXQxycnYtU6Do'
# # # #     api_client_Token='e9bk2MUfv_VHRBJsCVamROkB9bFtn_oTOb6-zI4tgzUNlmBcax5m2phWd0TOOpAY'
# # # #     main()
# # # if __name__ == "__main__":
# # #     main()

# ## --version 3.0 --##
# import os
# import csv
# import time
# import re
# import pandas as pd
# from datetime import datetime

# try:
#     import lyricsgenius
#     GENIUS_AVAILABLE = True
# except ImportError:
#     GENIUS_AVAILABLE = False

# # Constants
# OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "student_dataset.csv")
# MIN_SONGS = 25  # Updated to 100 songs as per assignment requirement

# def clean_lyrics(lyrics):
#     """Clean lyrics text to match the required format"""
#     if not lyrics:
#         return ""
    
#     # Remove contributor information at the beginning
#     lyrics = re.sub(r'^\d+\s+Contributors.*?Lyrics', '', lyrics, flags=re.IGNORECASE)
    
#     # Remove Genius-specific footers
#     lyrics = re.sub(r'Embed.*$', '', lyrics, flags=re.DOTALL | re.IGNORECASE)
    
#     # Remove section headers like [Verse], [Chorus], etc.
#     lyrics = re.sub(r'\[.*?\]', '', lyrics)
    
#     # Remove musical notations and movement names (for classical pieces)
#     movement_patterns = [
#         r'Allegro(\s+\w+)*', r'Adagio(\s+\w+)*', r'Largo(\s+\w+)*', 
#         r'Presto(\s+\w+)*', r'Andante(\s+\w+)*', r'Sonata', 
#         r'Movement', r'Opus', r'No\.\s*\d+'
#     ]
#     for pattern in movement_patterns:
#         lyrics = re.sub(pattern, '', lyrics, flags=re.IGNORECASE)
    
#     # Remove instrumental indicators
#     lyrics = re.sub(r'\{Instrumental\}', '', lyrics)
    
#     # Remove line numbers
#     lyrics = re.sub(r'^\d+[\.\)]\s*', '', lyrics, flags=re.MULTILINE)
    
#     # Convert newlines to spaces
#     lyrics = re.sub(r'\n', ' ', lyrics)
    
#     # Remove extra whitespace
#     lyrics = re.sub(r'\s+', ' ', lyrics).strip()
    
#     # Remove punctuation (optional - to match example format)
#     lyrics = re.sub(r'[^\w\s]', '', lyrics)
    
#     # Convert to lowercase (optional - to match example format)
#     lyrics = lyrics.lower()
    
#     return lyrics

# def manual_entry():
#     """Manually enter song information"""
#     print("\n=== Manual Song Entry ===")
    
#     # Create data directory if it doesn't exist
#     os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
#     # Get genre name
#     genre = input("Enter the genre name for all songs: ").strip().lower()
#     print(f"Genre set to: {genre}")
#     print(f"You need to enter at least {MIN_SONGS} songs. Press Ctrl+C to stop at any time.")
    
#     # Check if file exists and load existing data
#     songs = []
#     if os.path.exists(OUTPUT_FILE):
#         try:
#             df = pd.read_csv(OUTPUT_FILE)
#             songs = df.to_dict('records')
#             print(f"Loaded {len(songs)} existing songs from {OUTPUT_FILE}")
#         except Exception as e:
#             print(f"Error loading existing data: {e}")
    
#     # Start entry
#     count = len(songs)
#     try:
#         while count < MIN_SONGS:
#             print(f"\nSong {count+1}/{MIN_SONGS}")
#             artist = input("Artist name: ").strip()
#             track = input("Track name: ").strip()
            
#             while True:
#                 year = input("Release year (YYYY): ").strip()
#                 if re.match(r'^\d{4}$', year):
#                     break
#                 print("Invalid year format. Please enter a 4-digit year.")
            
#             print("Enter lyrics (type 'END' on a new line when finished):")
#             lyrics_lines = []
#             while True:
#                 line = input()
#                 if line.strip() == "END":
#                     break
#                 lyrics_lines.append(line)
            
#             lyrics = "\n".join(lyrics_lines)
#             lyrics = clean_lyrics(lyrics)
            
#             songs.append({
#                 "artist_name": artist,
#                 "track_name": track,
#                 "release_date": int(year),
#                 "genre": genre,
#                 "lyrics": lyrics
#             })
            
#             # Save after each song
#             df = pd.DataFrame(songs)
#             df.to_csv(OUTPUT_FILE, index=False)
#             print(f"Song added. Progress: {len(songs)}/{MIN_SONGS}")
            
#             count += 1
    
#     except KeyboardInterrupt:
#         print("\nEntry stopped by user.")
    
#     # Final save
#     df = pd.DataFrame(songs)
#     df.to_csv(OUTPUT_FILE, index=False)
    
#     print(f"\nTotal songs collected: {len(songs)}")
#     print(f"Data saved to {OUTPUT_FILE}")

# def api_collection():
#     """Collect data using LyricsGenius API"""
#     if not GENIUS_AVAILABLE:
#         print("LyricsGenius package not installed. Install it with:")
#         print("pip install lyricsgenius")
#         return
    
#     print("\n=== Genius API Song Collection ===")
    
#     # Create data directory if it doesn't exist
#     os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
#     # Get API key
#     api_key = input("Enter your Genius API key: ").strip()
    
#     # Initialize Genius API
#     try:
#         genius = lyricsgenius.Genius(api_key)
#         genius.verbose = False
#         # Skip non-songs to avoid instrumentals, etc.
#         genius.skip_non_songs = True
#         # Remove section headers
#         genius.remove_section_headers = True
#     except Exception as e:
#         print(f"Error initializing Genius API: {e}")
#         return
    
#     # Get genre name
#     genre = input("Enter the genre name for all songs: ").strip().lower()
#     print(f"Genre set to: {genre}")
    
#     # Get artists
#     print("Enter artist names known for this genre (one per line, type 'END' when finished):")
#     artists = []
#     while True:
#         artist = input().strip()
#         if artist == "END":
#             break
#         artists.append(artist)
    
#     if not artists:
#         print("No artists entered. Exiting.")
#         return
    
#     print(f"Collecting songs for {len(artists)} artists in the {genre} genre.")
    
#     # Check if file exists and load existing data
#     songs = []
#     if os.path.exists(OUTPUT_FILE):
#         try:
#             df = pd.read_csv(OUTPUT_FILE)
#             songs = df.to_dict('records')
#             print(f"Loaded {len(songs)} existing songs from {OUTPUT_FILE}")
#         except Exception as e:
#             print(f"Error loading existing data: {e}")
    
#     # Collect songs
#     for artist_name in artists:
#         print(f"\nSearching for songs by {artist_name}...")
        
#         try:
#             # Get artist
#             artist = genius.search_artist(artist_name, max_songs=50, sort="popularity")
            
#             if not artist:
#                 print(f"Could not find artist: {artist_name}")
#                 continue
            
#             # Process songs
#             for song in artist.songs:
#                 if len(songs) >= MIN_SONGS:
#                     break
                
#                 # Skip if song has no lyrics
#                 if not song.lyrics:
#                     continue
                
#                 # Skip if lyrics are too short (likely instrumental)
#                 if len(song.lyrics.split()) < 20:
#                     print(f"Skipping {song.title} - lyrics too short (likely instrumental)")
#                     continue
                
#                 # Get release year
#                 try:
#                     year_str = song.year
#                     if year_str and re.match(r'^\d{4}$', year_str):
#                         year = int(year_str)
#                     else:
#                         # Use current year as fallback
#                         year = datetime.now().year
#                 except:
#                     year = datetime.now().year
                
#                 # Clean lyrics
#                 lyrics = clean_lyrics(song.lyrics)
                
#                 # Skip if cleaned lyrics are too short
#                 if len(lyrics.split()) < 20:
#                     print(f"Skipping {song.title} - cleaned lyrics too short")
#                     continue
                
#                 # Add song to collection
#                 songs.append({
#                     "artist_name": artist_name,
#                     "track_name": song.title,
#                     "release_date": year,
#                     "genre": genre,
#                     "lyrics": lyrics
#                 })
                
#                 print(f"Added: {song.title} - Progress: {len(songs)}/{MIN_SONGS}")
                
#                 # Save after each song
#                 df = pd.DataFrame(songs)
#                 df.to_csv(OUTPUT_FILE, index=False)
                
#                 # Respect API rate limits
#                 time.sleep(1)
        
#         except Exception as e:
#             print(f"Error processing artist {artist_name}: {e}")
    
#     # Final report
#     print(f"\nTotal songs collected: {len(songs)}")
#     print(f"Data saved to {OUTPUT_FILE}")
    
#     if len(songs) < MIN_SONGS:
#         print(f"Warning: Only collected {len(songs)} songs, which is less than the minimum required ({MIN_SONGS}).")
#         print("Consider adding more artists or using manual entry to reach the minimum.")

# def merge_datasets():
#     """Merge student dataset with the Mendeley dataset"""
#     student_file = OUTPUT_FILE
#     mendeley_file = os.path.join(os.path.dirname(OUTPUT_FILE), "mendeley_dataset.csv")
#     merged_file = os.path.join(os.path.dirname(OUTPUT_FILE), "merged_dataset.csv")
    
#     if not os.path.exists(student_file):
#         print(f"Error: Student dataset not found at {student_file}")
#         return
    
#     if not os.path.exists(mendeley_file):
#         print(f"Error: Mendeley dataset not found at {mendeley_file}")
#         return
    
#     try:
#         # Load datasets
#         student_df = pd.read_csv(student_file)
#         mendeley_df = pd.read_csv(mendeley_file)
        
#         # Select required columns from Mendeley dataset
#         mendeley_df = mendeley_df[["artist_name", "track_name", "release_date", "genre", "lyrics"]]
        
#         # Concatenate datasets
#         merged_df = pd.concat([mendeley_df, student_df], ignore_index=True)
        
#         # Save merged dataset
#         merged_df.to_csv(merged_file, index=False)
        
#         print(f"Successfully merged datasets:")
#         print(f"- Mendeley dataset: {len(mendeley_df)} songs")
#         print(f"- Student dataset: {len(student_df)} songs")
#         print(f"- Merged dataset: {len(merged_df)} songs")
#         print(f"Merged dataset saved to {merged_file}")
    
#     except Exception as e:
#         print(f"Error merging datasets: {e}")

# def main():
#     """Main function to run the script"""
#     print("\n=== Student Dataset Generator ===")
#     print("This script helps you create a dataset for an additional music genre.")
    
#     while True:
#         print("\nChoose an option:")
#         print("1. Manual song entry")
#         print("2. Collect songs using Genius API")
#         print("3. Merge with Mendeley dataset")
#         print("4. Exit")
        
#         choice = input("Enter your choice (1-4): ").strip()
        
#         if choice == "1":
#             manual_entry()
#         elif choice == "2":
#             api_collection()
#         elif choice == "3":
#             merge_datasets()
#         elif choice == "4":
#             print("Exiting. Goodbye!")
#             break
#         else:
#             print("Invalid choice. Please enter a number between 1 and 4.")

# if __name__ == "__main__":
#     main()
# # # # if __name__ == "__main__":
# # # #     # Q3XZOS7nlFx-3nLwM1-uf_Jr68XSQx7DggOc43F_QcK2aOC0qxjTXQxycnYtU6Do
# # # #     api_client_ID='Q3XZOS7nlFx-3nLwM1-uf_Jr68XSQx7DggOc43F_QcK2aOC0qxjTXQxycnYtU6Do'
# # # #     api_client_Token='e9bk2MUfv_VHRBJsCVamROkB9bFtn_oTOb6-zI4tgzUNlmBcax5m2phWd0TOOpAY'
# # # #     main()
# # # if __name__ == "__main__":
# # #     main()
# # Antonio Vivaldi
# # Frédéric Chopin
# # Johannes Brahms
# # George Frideric Handel
# # Sergei Rachmaninoff
# # Maurice Ravel
# # Claude Debussy
# # Igor Stravinsky
# # Joseph Haydn
# # Camille Saint-Saëns
# # Domenico Scarlatti
# # END


import pandas as pd

# File paths
mendeley_dataset_path = "/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/data/mendeley_music_dataset.csv"
student_dataset_path = "/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/student_dataset.csv"

# Read the datasets directly from the file paths
mendeley_df = pd.read_csv(mendeley_dataset_path)
student_df = pd.read_csv(student_dataset_path)

# Keep only the required columns from the Mendeley dataset
mendeley_df = mendeley_df[['artist_name', 'track_name', 'release_date', 'genre', 'lyrics']]

# Make sure the student dataset has all required columns
required_columns = ['artist_name', 'track_name', 'release_date', 'genre', 'lyrics']
for col in required_columns:
    if col not in student_df.columns:
        print(f"Warning: '{col}' column missing from student dataset!")

# Select only the required columns from student dataset (in case it has extra columns)
student_df = student_df[required_columns]

# Merge the datasets
merged_df = pd.concat([mendeley_df, student_df], ignore_index=True)

# Display information about the merged dataset
print("Mendeley Dataset Shape:", mendeley_df.shape)
print("Student Dataset Shape:", student_df.shape)
print("Merged Dataset Shape:", merged_df.shape)
print("\nFirst 5 rows of the merged dataset:")
print(merged_df.head())

# Check for any missing values
print("\nMissing values in merged dataset:")
print(merged_df.isnull().sum())

# Optionally save the merged dataset to a CSV file
output_path = "/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/merged_music_dataset.csv"
merged_df.to_csv(output_path, index=False)
print(f"\nMerged dataset has been saved to '{output_path}'")