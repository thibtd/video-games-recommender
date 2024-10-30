import pandas as pd
from app import stream_data


def test_stream_data():
    # Create a sample DataFrame
    data = pd.DataFrame(
        {
            "name": ["Game 1", "Game 2"],
            "rating": [9.0, 8.5],
            "genres": ["Action, Adventure", "Sci-Fi, Thriller"],
            "plot": ["Plot 1", "Plot 2"],
            "url": ["http://example.com/game1", "http://example.com/game2"],
        }
    )

    # Call the stream_data function
    generator = stream_data(data)

    # Collect the output
    output = "".join([word for word in generator])

    # Expected output
    expected_output = (
        "1. **Game 1** \n\r Rating: *9.0*, Genres: *Action, Adventure* \n\n Plot 1 \n\n Check it out on IMDb: *http://example.com/game1*\n\r"
        "2. **Game 2** \n\r Rating: *8.5*, Genres: *Sci-Fi, Thriller* \n\n Plot 2 \n\n Check it out on IMDb: *http://example.com/game2*\n\r"
    )
    # Split output and expected_output into lines and strip trailing whitespace
    output_lines = [line.rstrip() for line in output.split("\n")]
    expected_lines = [line.rstrip() for line in expected_output.split("\n")]
    # Assert the output is as expected line by line
    assert output_lines == expected_lines
