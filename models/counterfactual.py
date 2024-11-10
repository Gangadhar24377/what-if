# models/counterfactual.py
import torch
import torch.nn.functional as F

class CounterfactualGenerator:
    def __init__(self, classifier, learning_rate=0.01, num_iterations=1000):
        self.classifier = classifier
        self.lr = learning_rate
        self.num_iterations = num_iterations
        
    def generate(self, image, target_class, lambda_reg=0.1):
        """Generate counterfactual explanation."""
        # Initialize counterfactual as copy of original image
        counterfactual = image.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([counterfactual], lr=self.lr)
        
        original_features = self.classifier.get_features(image)
        
        for i in range(self.num_iterations):
            optimizer.zero_grad()
            
            # Classification loss (push towards target class)
            outputs = self.classifier(counterfactual)
            target = torch.tensor([target_class])
            cls_loss = F.cross_entropy(outputs, target)
            
            # Feature similarity loss (maintain similarity to original)
            cf_features = self.classifier.get_features(counterfactual)
            feat_loss = F.mse_loss(cf_features, original_features)
            
            # L2 regularization (minimal perturbation)
            l2_loss = F.mse_loss(counterfactual, image)
            
            # Total loss
            loss = cls_loss + lambda_reg * (feat_loss + l2_loss)
            loss.backward()
            optimizer.step()
            
            # Clip values to valid image range
            counterfactual.data.clamp_(0, 1)
            
        return counterfactual.detach()