#counterfactual.py
import torch
import torch.nn.functional as F

class CounterfactualGenerator:
    def __init__(self, classifier, learning_rate=0.01, num_iterations=100):
        self.classifier = classifier
        self.lr = learning_rate
        self.num_iterations = num_iterations
        
    def generate(self, image, target_class, class_labels, lambda_reg=0.1):
        """Generate counterfactual explanation."""
        # Initialize counterfactual as copy of original image
        counterfactual = image.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([counterfactual], lr=self.lr)
        
        original_features = self.classifier.get_features(image)
        
        # Find the index of the target_class in the class_labels list
        target_class_index = class_labels.index(target_class)
        
        for i in range(self.num_iterations):
            optimizer.zero_grad()
            
            outputs = self.classifier(counterfactual)
            target = torch.tensor([target_class_index])  # Use target_class_index
            cls_loss = F.cross_entropy(outputs, target)
            
            # Feature similarity loss (maintain similarity to original)
            cf_features = self.classifier.get_features(counterfactual)
            feat_loss = F.mse_loss(cf_features, original_features)
            
            # L2 regularization (minimal perturbation)
            l2_loss = F.mse_loss(counterfactual, image)
            
            # Total loss
            loss = cls_loss + lambda_reg * (feat_loss + l2_loss)
            loss.backward(retain_graph=True)
            optimizer.step()
            
            # Clip values to valid image range
            counterfactual.data.clamp_(0, 1)
            
        return counterfactual.detach()