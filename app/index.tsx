import React, { useState } from "react";
import { Platform } from "react-native";
import { Text, View, StyleSheet, Button, ActivityIndicator, Alert } from "react-native";
import * as ImagePicker from "expo-image-picker";
import { Image } from "expo-image";

const PlaceholderImage = require("@/assets/images/tomatebg.jpg");

export default function App() {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const takePhoto = async () => {
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();

    if (!permissionResult.granted) {
      Alert.alert("Permission refusée", "Autorisez l'accès à la caméra.");
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      const uri = result.assets[0].uri;
      setImageUri(uri);
      sendToAPI(uri);
    }
  };

  const sendToAPI = async (uri: string) => {
    setLoading(true);
    setPrediction(null);
  
    const formData = new FormData();
  
    // Web support : créer un vrai fichier depuis fetch/Blob si besoin
    const filename = "leaf.jpg";
    const type = "image/jpeg";
  
    if (Platform.OS === "web") {
      // Pour le Web : on récupère l’image en tant que Blob via fetch
      const response = await fetch(uri);
      const blob = await response.blob();
  
      formData.append("file", new File([blob], filename, { type }));
    } else {
      // Pour iOS / Android
      formData.append("file", {
        uri,
        name: filename,
        type,
      } as any);
    }
  
    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
        headers: {
          Accept: "application/json",
          // ❌ PAS de 'Content-Type', c’est `fetch` qui s’en charge
        },
      });
  
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error("Erreur API: " + errorText);
      }
  
      const data = await response.json();
      console.log("Réponse API:", data);
      setPrediction(data.result?.pred || data.prediction || "Réponse inconnue");
    } catch (error: unknown) {
      if (error instanceof Error) {
        Alert.alert("Erreur", error.message);
        console.error("API Error:", error);
      } else {
        Alert.alert("Erreur inconnue", "Une erreur est survenue.");
        console.error("Erreur inconnue", error);
      }
    } finally {
      setLoading(false);
    }
  };
  

  return (
    <View style={styles.container}>
      <Image source={PlaceholderImage} style={styles.backgroundImage} />

      {imageUri && <Image source={{ uri: imageUri }} style={styles.image} />}
      <Button title="📸 Prendre une photo" onPress={takePhoto} />

      {loading && (
        <ActivityIndicator size="large" color="#00ff00" style={{ marginTop: 20 }} />
      )}

      {prediction && <Text style={styles.prediction}>Résultat : {prediction}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#25292e",
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
  },
  backgroundImage: {
    ...StyleSheet.absoluteFillObject,
    resizeMode: "cover",
  },
  image: {
    width: 300,
    height: 400,
    borderRadius: 10,
    marginBottom: 20,
  },
  prediction: {
    color: "#fff",
    fontSize: 20,
    marginTop: 20,
  },
});
