# Projet CNN Accéléré GPU avec Numba  
**Auteur : Khalil Ghouddan**

---

## 📌 Contexte du Projet

Ce projet consiste à implémenter un réseau de neurones convolutif (CNN) capable de reconnaître les chiffres manuscrits (0 à 9).  
L’implémentation de départ provient des excellents articles de Victor Zhou :

- https://victorzhou.com/blog/intro-to-cnns-part-1/  
- https://victorzhou.com/blog/intro-to-cnns-part-2/

Le code d'origine utilise uniquement **Python en mono-thread**, **sans GPU**, et avec très peu de bibliothèques externes.

L’objectif principal de ce projet est d’optimiser ce CNN en utilisant **Numba** afin de tirer parti du **GPU** pour accélérer les calculs.

---

## 🎯 Objectif du Projet

Le projet part du code source disponible ici :  
👉 https://github.com/fabricehuet/cnn-python  

Vous devez modifier ce code pour :

1. **Exécuter des parties critiques sur GPU avec Numba**  
2. **Accélérer le modèle par rapport à la version CPU**  
3. **Créer un script performance bench.py** pour comparer CPU vs GPU  
4. **Créer un script analyze.py** pour reconnaître plusieurs chiffres dans une image JPG  
5. **Rédiger un rapport Readme.md (celui-ci)** avec toutes les explications demandées

---

## 🛠️ Modifications pour l’Exécution GPU

### ✔️ Pourquoi utiliser Numba ?

Numba permet de compiler du code Python en machine code via LLVM et CUDA.  
Cela permet d’exécuter certaines fonctions directement sur GPU avec de grandes performances.

### ✔️ Parties du code modifiées

Voici les parties du CNN qui ont été adaptées pour s'exécuter sur GPU :

#### 1️⃣ **La couche Convolution (Conv3x3)**  
- Initialement, la convolution utilisait des boucles Python imbriquées → très lent.
- Le nouveau code utilise un kernel CUDA avec Numba :
  ```python
  @cuda.jit
  def conv_gpu(image, kernel, output):
      i, j = cuda.grid(2)
      if i < output.shape[0] and j < output.shape[1]:
          val = 0.0
          for ki in range(3):
              for kj in range(3):
                  val += image[i+ki, j+kj] * kernel[ki, kj]
          output[i, j] = val
