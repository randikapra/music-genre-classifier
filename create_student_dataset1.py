'''import lyricsgenius
import pandas as pd
import time
import re
from datetime import datetime

# Initialize Genius API client
genius = lyricsgenius.Genius("UBCcFO_YPq_Ppvc-lpLVkyIaCyIicxeh9SivvTgKPZd2HFEVzA3PoOXRX1phKvNE")
genius.verbose = False  # Turn off status messages
genius.remove_section_headers = True  # Remove section headers (e.g. [Chorus]) from lyrics

# List of K-pop artists to search for
kpop_artists = [
    "BTS", "BLACKPINK", "TWICE", "EXO", "Red Velvet", "NCT", "SEVENTEEN", 
    "ITZY", "Stray Kids", "ATEEZ", "IU", "MAMAMOO", "GOT7", "TXT", 
    "ENHYPEN", "aespa", "NewJeans", "IVE", "G-IDLE", "BTOB", "MONSTA X", 
    "SHINee", "Girls' Generation", "Super Junior", "2NE1", "BigBang", 
    "Wonder Girls", "KARA", "SISTAR", "f(x)", "4Minute", "VIXX", "INFINITE",
    "BoA", "PSY", "TVXQ", "2PM", "WINNER", "iKON", "Pentagon", "Day6", 
    "ASTRO", "The Boyz", "TREASURE", "GFRIEND", "LOONA", "WJSN", "AOA",
    "Apink", "CLC", "SF9", "AB6IX", "ONEUS", "ONF", "(G)I-DLE", "Kep1er"
]

def extract_year(date_str):
    """Extract the year from a release date string"""
    if not date_str:
        return None
        
    # Try various formats
    for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%B %d, %Y', '%Y'):
        try:
            date_obj = datetime.strptime(date_str, fmt)
            return date_obj.year
        except ValueError:
            continue
            
    # If all formats fail, try to find a 4-digit number that could be a year
    year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
    if year_match:
        return int(year_match.group(0))
        
    return None

def clean_lyrics(lyrics):
    """Clean lyrics by removing metadata and unnecessary spaces"""
    if not lyrics:
        return ""
        
    # Remove any Genius-specific metadata that might be at the end
    lyrics = re.sub(r'\d+Embed', '', lyrics)
    lyrics = re.sub(r'You might also like', '', lyrics)
    
    # Remove additional line breaks and standardize spacing
    lyrics = re.sub(r'\n{3,}', '\n\n', lyrics)
    lyrics = lyrics.strip()
    
    return lyrics

# Function to collect songs from an artist
def collect_artist_songs(artist_name, max_songs=25):
    songs_data = []
    
    try:
        # Search for the artist
        artist = genius.search_artist(artist_name, max_songs=max_songs, sort="popularity")
        
        if not artist:
            print(f"Could not find artist: {artist_name}")
            return []
            
        print(f"Collecting songs for {artist_name}...")
        
        # Iterate through artist's songs
        for song in artist.songs:
            try:
                # Skip non-English or non-Korean songs (to focus on K-pop)
                if song.lyrics and (re.search(r'[가-힣]', song.lyrics) or song.language == "ko" or song.language == "en"):
                    # Get release date/year
                    release_year = None
                    
                    # Try to get release date from song metadata
                    if hasattr(song, 'release_date') and song.release_date:
                        release_year = extract_year(song.release_date)
                    
                    # If release_year is still None, try to get from album or other metadata
                    if not release_year and hasattr(song, 'album') and song.album and hasattr(song.album, 'release_date'):
                        release_year = extract_year(song.album.release_date)
                    
                    # Clean lyrics
                    clean_lyric = clean_lyrics(song.lyrics)
                    
                    # Create song data dictionary
                    song_data = {
                        'artist_name': artist_name,
                        'track_name': song.title,
                        'release_date': release_year,
                        'genre': 'K-pop',
                        'lyrics': clean_lyric
                    }
                    
                    songs_data.append(song_data)
                    print(f"Added: {song.title}")
                    
            except Exception as e:
                print(f"Error processing song {song.title}: {e}")
                continue
                
            # Sleep to avoid hitting rate limits
            time.sleep(1)
            
    except Exception as e:
        print(f"Error collecting songs for artist {artist_name}: {e}")
    
    return songs_data

# Collect songs
all_songs = []
songs_per_artist = 25  # Adjust as needed to reach 100+ songs

for artist in kpop_artists:
    songs = collect_artist_songs(artist, max_songs=songs_per_artist)
    all_songs.extend(songs)
    
    # Break if we have enough songs
    if len(all_songs) >= 1250:  # Aiming for a bit over 100 to account for potential duplicates
        break
        
    # Sleep between artists to avoid hitting rate limits
    time.sleep(3)

# Create DataFrame
kpop_df = pd.DataFrame(all_songs)

# Drop duplicates and rows with missing crucial information
kpop_df = kpop_df.drop_duplicates(subset=['artist_name', 'track_name'])
kpop_df = kpop_df.dropna(subset=['track_name', 'lyrics'])

# Fill missing release dates with a placeholder
kpop_df['release_date'] = kpop_df['release_date'].fillna(0).astype(int)

# Make sure we have at least 100 songs
if len(kpop_df) < 1000:
    print(f"Warning: Only collected {len(kpop_df)} valid songs. Need at least 100.")
else:
    print(f"Successfully collected {len(kpop_df)} K-pop songs!")

# Save to CSV
kpop_df.to_csv("/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/data/student_dataset.csv", index=False)
print("Dataset saved as 'kpop_dataset.csv'")

# Display the first few rows
print(kpop_df.head())

# Dataset statistics
print(f"\nDataset Statistics:")
print(f"Total songs: {len(kpop_df)}")
print(f"Unique artists: {kpop_df['artist_name'].nunique()}")
print(f"Years covered: {kpop_df['release_date'].min()} to {kpop_df['release_date'].max()}")


'''

