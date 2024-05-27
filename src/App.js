import React, { useState } from "react";
import axios from "axios";
import {
  Container,
  Typography,
  Box,
  Button,
  CircularProgress,
  Alert,
  TextField,
  Card,
  CardContent,
  CardHeader,
  AppBar, 
  Toolbar, 
  IconButton 
} from "@mui/material";
import { makeStyles } from "@mui/styles";
import "./App.css";
import TwitterIcon from '@mui/icons-material/Twitter';
import EventIcon from "@mui/icons-material/Event";
import ParticleBackground from "./ParticleBackground.js";

const useStyles = makeStyles((theme) => ({
  container: {
    marginTop: '120px',
    textAlign: 'center',
    position: 'relative',
    zIndex: 2,
  },
  input: {
    display: 'none',
  },
  button: {
    margin: '20px 0',
  },
  resultBox: {
    margin: '20px 0',
    textAlign: 'left',
  },
  eventCard: {
    margin: '10px 0',
    borderRadius: '10px',
    boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
    transition: 'transform 0.2s',
    '&:hover': {
      transform: 'scale(1.02)',
    },
  },
  eventHeader: {
    backgroundColor: '#f5f5f5',
    borderBottom: '1px solid #ddd',
  },
  eventContent: {
    textAlign: 'left',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '20px',
  },
  icon: {
    marginRight: '10px',
    verticalAlign: 'middle',
  },
  appBar: {
    position: 'fixed',
    backgroundColor: '#1DA1F2',
  },
  toolbar: {
    display: 'flex',
    justifyContent: 'space-between',
  },
  twitterIcon: {
    color: '#fff',
  },
}));

function App() {
  const classes = useStyles();
  const [file, setFile] = useState(null);
  const [numEvents, setNumEvents] = useState(5); 
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResults(null);
    setError(null);
  };

  const handleNumEventsChange = (e) => {
    setNumEvents(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    formData.append("numEvents", numEvents);

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        "http://localhost:5000/upload",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      setResults(response.data);
    } catch (error) {
      console.error(error);
      setError("Error uploading file. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <ParticleBackground />
      <AppBar className={classes.appBar}>
        <Toolbar className={classes.toolbar}>
          <IconButton edge="start" color="inherit" aria-label="menu">
            <TwitterIcon className={classes.twitterIcon} />
          </IconButton>
        </Toolbar>
      </AppBar>
      <Container className={classes.container} >
        <Typography variant="h4" gutterBottom>
          <EventIcon className={classes.icon} /> Twitter Event Detection
        </Typography>
        <form onSubmit={handleSubmit}>
          <input
            accept=".json"
            className={classes.input}
            id="contained-button-file"
            type="file"
            onChange={handleFileChange}
          />
          <label htmlFor="contained-button-file">
            <Button variant="contained" color="primary" component="span">
              Upload File
            </Button>
          </label>
          <TextField
            type="number"
            value={numEvents}
            onChange={handleNumEventsChange}
            label="Number of Events"
            variant="outlined"
            margin="normal"
            inputProps={{ min: 1 }}
          />
          <Button
            type="submit"
            variant="contained"
            color="secondary"
            className={classes.button}
            disabled={!file}
          >
            Detect Events
          </Button>
        </form>
        {loading && (
          <div>
            {" "}
            <div style={{ height: "5em" }}></div> <CircularProgress />
          </div>
        )}
        {error && <Alert severity="error">{error}</Alert>}
        {results && (
          <Box className={classes.resultBox}>
            <Typography variant="h5" gutterBottom>
              Detected Events
            </Typography>
            {results.events.map((event, index) => (
              <Card key={index} className={classes.eventCard}>
                <CardHeader
                  title={`Event ${index + 1}`}
                  className={classes.eventHeader}
                />
                <CardContent className={classes.eventContent}>
                  <Typography variant="body1">
                    <strong>Top Keywords:</strong> {event.keywords.join(", ")}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    <strong>Representative Tweet:</strong> {event.tweet}
                  </Typography>
                </CardContent>
              </Card>
            ))}
          </Box>
        )}
      </Container>
    </div>
  );
}

export default App;
