    # save our tensor_images 
    torch.save(tensor_images, uncertainty_root + '/dropblock_tests.pt')
    
    # save our numpy data
    losses.tofile(os.path.join(uncertainty_root, 'avged_losses.txt'), sep = '\n', format = '%ls')
    
    
    

    
    
    
    
    
    
    
        
