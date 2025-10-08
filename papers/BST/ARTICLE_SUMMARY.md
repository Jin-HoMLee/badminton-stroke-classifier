# **BST: Badminton Stroke-type Transformer for Skeleton-based Action Recognition in Racket Sports**

**Authors:** Jing-Yuan Chang, National Tsing Hua University

---

## **Overview**

This paper presents the Badminton Stroke-type Transformer (BST), a novel deep learning model for skeleton-based action recognition in racket sports, with a focus on badminton. By leveraging player pose, position, and shuttlecock trajectory information, BST substantially advances the accuracy and robustness of stroke-type classification in broadcast sports videos. The approach is validated across leading badminton and tennis datasets, outperforming state-of-the-art models.

---

## **Motivation and Challenges**

- Badminton’s extreme ball speed and dynamic player movements pose considerable challenges for computer vision tasks such as player identification, shuttlecock tracking, court detection, and stroke classification.
- Traditional human action recognition (HAR) and skeleton-based action recognition (SAR) models struggle with subtle, rapid, and brief racket sports actions, often misclassifying similar strokes due to limited context and lack of interaction modeling.
- Existing models rarely exploit the shuttlecock trajectory as a core input, though it is the most objective and informative signal for distinguishing player actions.

---

## **Key Contributions**

1. **Novel Video Clipping Strategy:**
   - Segments match videos into clips that contain highly relevant frames for each stroke, capturing not just the player’s action but also critical context from opponent movements and shuttlecock trajectory.
   - The strategy ensures each clip includes the second half of the previous stroke, the full target stroke, and the first half of the next stroke, enabling richer temporal and interactional modeling.

2. **BST Architecture:**
   - A Transformer-based model that incorporates:
     - **BST-0 Backbone:** Dedicated temporal convolutional networks for pose and shuttlecock trajectory, followed by Transformer encoders and cross-attention modules.
     - **Pose Position Fusion (PPF):** Fuses pose with court position information.
     - **Clean Gate (CG):** Denoises shuttlecock trajectory by filtering out irrelevant opponent-induced motion.
     - **Aim Player (AP):** Weights player contributions based on correlation to shuttlecock trajectory.
     - **Combined Variants:** (BST-CG, BST-AP, BST-CG-AP) for further accuracy gains.

3. **Empirical Validation:**
   - Demonstrates state-of-the-art results on three benchmark datasets:
     - **ShuttleSet** (largest badminton dataset), **BadmintonDB**, and **TenniSet** (tennis).
   - Shows strong generalization, outperforming previous models even with limited training data.

4. **Supplemental Analyses:**
   - Training speed: BST models converge faster and are computationally efficient.
   - 2D vs 3D joints: 2D pose estimation is more reliable for racket sports, as 3D estimators trained on general datasets often produce inaccurate joints.
   - Detailed dataset splits, class definitions, and hyperparameters provided for reproducibility.

---

## **Related Work**

- **Badminton Video Analysis:** Prior works have focused on end-to-end frameworks for player attribute tagging, stroke classification (typically for one player), serve detection, and shuttlecock tracking (TrackNet, MonoTrack).
- **Skeleton-based Action Recognition:** Advances in graph convolutional networks (GCN) and Transformers (ST-GCN, BlockGCN, SkateFormer, ProtoGCN, TemPose) have improved HAR/SAR, but were not specifically tailored for the nuances and interactions in racket sports.
- BST builds upon and surpasses these approaches by integrating ball trajectory as a primary signal and modeling player interactions more effectively.

---

## **Methodology**

1. **Video Clipping:**  
   - Clips are centered around hit frames, using a combination of fixed-width, pose-based, and trajectory-based strategies for optimal context.
   - Clipping parameters are carefully chosen to balance inclusion of relevant motion and exclusion of noise.

2. **Input Extraction:**  
   - Player poses and positions are extracted using state-of-the-art pose estimation tools (MMPose, RTMPose, MotionBERT).
   - Shuttlecock trajectories are tracked using advanced models (TrackNetV3, MonoTrack), with court line detection to filter out irrelevant subjects.

3. **Model Structure:**  
   - BST processes sequences using Temporal Convolutional Networks and Transformer Encoders, leveraging Cross Transformer layers for pose-trajectory fusion.
   - Additional modules (PPF, CG, AP) enhance representation learning and decision-making.

---

## **Experiments and Results**

- **Datasets:**  
  - ShuttleSet (25 and 35 class versions, fine and merged granularity), BadmintonDB (18 classes), TenniSet (6 classes).
- **Metrics:**  
  - Accuracy, Macro-F1, Min-F1, Top-2 Accuracy.
- **Findings:**  
  - BST models consistently outperform prior state-of-the-art, especially when incorporating shuttlecock trajectory and player position inputs.
  - The novel clipping strategy further boosts performance, particularly in high-class-granularity settings.
  - BST demonstrates superior generalization when trained on reduced data, making it robust in low-data regimes.
  - 2D joints are preferable over 3D for racket sports due to better accuracy and representation fidelity.

---

## **Discussion & Limitations**

- **Classification Difficulty:**  
  - High granularity (many fine-grained classes) introduces confusion among similar strokes; BST mitigates this but some confusion remains.
- **Pose Inputs:**  
  - 3D joints derived from general datasets can be misleading for specialized badminton actions; 2D joints provide better results.
- **Dependence on Tracking:**  
  - The method’s success is contingent on accurate hit frame detection and shuttlecock trajectory tracking, though these tools are becoming increasingly reliable.
- **Future Directions:**  
  - Extending BST to other racket sports (e.g., table tennis, doubles) and improving 3D ball tracking are promising areas for future research.

---

## **Conclusion**

BST introduces a powerful combination of innovative video clipping, cross-modal Transformer architecture, and specialized modules, resulting in state-of-the-art badminton stroke-type classification. Its emphasis on shuttlecock trajectory information and player interaction modeling sets a new benchmark for skeleton-based action recognition in racket sports.

---

## **References & Implementation Resources**

- **Key open-source models:**  
  - TrackNetV3 ([GitHub](https://github.com/alenzenx/TrackNetV3))
  - MMPose ([GitHub](https://github.com/open-mmlab/mmpose))
  - TennisCourtDetector ([GitHub](https://github.com/yastrebksv/TennisCourtDetector))
- **Datasets:**  
  - ShuttleSet, BadmintonDB, TenniSet (details and splits provided in the supplement).

**For implementation guidance or code searches, the referenced tools and datasets provide a strong starting point.**

---